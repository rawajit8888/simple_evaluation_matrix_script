const modules = global.modules;
const { CustomRPABase, markerService, util, cacheService, ExcelApi } = modules;
const { chromium } = modules.require('playwright');
const path = require('path');
const fs = require('fs');
const XLSX = modules.require('xlsx');
const InteractiveAIConnector = require('./interactive_ai_inference_connector_test_prompt.js');
const PrecisionccConnector = require('./precision');
const replyToCustomerConnector = require('./replyCrm');
const PiiDetectorConnector = require('./piiDetector');
const { spawn, exec } = require('child_process');

class webRPAScript extends CustomRPABase {
    async process() {
        let self = this;
        let rslt;
        let $ = self.$;
        let rpaParams = this.rpaParams;
        let crmUserId = rpaParams.userid;
        let crmPassword = rpaParams.password;
        let crmSite = rpaParams.crmSite;
        let crmBaseUrl = rpaParams.crmBaseUrl;
        let filePath = rpaParams.filePath;
        let email_masterList_sheet = "List of email termed escalation"
        let CaseResultFilePath = rpaParams.CaseResultFilePath;
        let caseSheetName = 'Case Result Data';
        let bucket = rpaParams.bucket;
        let bucketSelectOptionId = rpaParams.bucketSelectOptionId.toString();
        let buss_id = $.buss_id;
        let sortBy = rpaParams.sortBy;//'Asc'; 
        let browser;
        try {
            //create folder for caseData Output 
            //create folder to save output log and returns path of output log
            CaseResultFilePath = this.getDatedExcelPath(CaseResultFilePath, "CsLog.xlsx");

            //get master Email from sheet
            this.masterList = await this.getMasterEmail(filePath, email_masterList_sheet);

            //agent all agents from sheet
            let agentSheetName = "Uat agents";
            this.agents = await this.loadAgentsFromExcel(filePath, agentSheetName);

            // Get parameters of AI inference from cacheservice
            rslt = await cacheService.getRecord('ai_inference__crm_parameters');
            let { config_dat: { params } } = rslt.data;

            // 1. Launch browser
            browser = await chromium.launch({ headless: false });
            const context = await browser.newContext();
            const page = await context.newPage();

            // 2. Login
            await self.loginToCRM(page, crmUserId, crmPassword, crmSite);

            // 3. Navigate to the bucket (e.g., Mail Desk = 80,connect =1001593,Grievance=160,nri=161)

            await self.navigateToBucket(page, bucketSelectOptionId);

            // 4. Get all case links
            let caseData = await this.getAllCustcarewebCaseLinks(page, sortBy);
            caseData = caseData.slice(0, 5);

            // 5. For each case, process it
            for (const c of caseData) {
                let clientType = 'Customer';
                let lastCaseId = await this.getLastCaseId();
                if (+c.value == lastCaseId) continue;
                // if (c.value == "05953266") continue;

                // Open case and get email body
                // Extract To/CC emails and 
                const { subject, from, toEmails, ccEmails, bodyText: emailBodyText, casePage } = await this.extractCustomerEmails(context, crmBaseUrl, c.href);

                //------------------------------------------------------
                //------------Pii Data----------------------------------

                // const piiConnector = new PiiDetectorConnector();
                // piiConnector.data = { emailBodyText }
                // piiConnector.scriptParams = {};
                // const { redactedText, piiMap } = await piiConnector.process();
                // console.log(redactedText);

               //masking email boy data
                let maskedData = await this.maskEmailPII(emailBodyText);
                console.log('Masked Text:' + maskedData.masked);
                // console.log('Entity Map:' + maskedData.entity_map);

                // check master list
                const allCaseEmails = [...toEmails, ...ccEmails, from].map(e => e.toLowerCase());
                const isSpecialCase = this.masterList.some(masterEmail =>
                    allCaseEmails.includes(masterEmail.toLowerCase())
                );
                // click edit button
                await this.clickEditButton(casePage);

                // Main caller function for each case
                let processCaseData =
                    { maskedData, subject, page, casePage, emailBodyText, from, toEmails, ccEmails, isSpecialCase, clientType, c, params, context, rpaParams, bucket, CaseResultFilePath, caseSheetName }
                const caseDataobj = await this.processCaseByTag(processCaseData);
                if (caseDataobj.rc != 0) {
                    await this.appendObjectToExcel(CaseResultFilePath, caseSheetName, caseDataobj)
                };
            }
            await browser.close();
            return { rc: 0 };
        } catch (err) {
            console.error(err.message);
            console.error(err.stack);
            await browser.close();
            return { rc: 1, msg: err.message };
        }

    }

    // Main caller for each case
    async processCaseByTag({ maskedData, subject, page, casePage, emailBodyText, from, toEmails, ccEmails, isSpecialCase, clientType, c, params, context, rpaParams, bucket, CaseResultFilePath, caseSheetName }) {
        let assignedAgent = null;
        let precisionData = null;
        const inputData = {
            crypto_pwd: params.crypto_pwd.value,
            crypto_salt: params.crypto_salt.value,
            case_id: c.value,
            subject,
            email_message: maskedData.masked,
            client_id: params.client_id.value,
            client_secret: params.client_secret.value,
            host: params.host.value,
            auth_token: params.auth_token.value,
            serviceId: params.serviceId.value,
            serviceId_entity_extraction: params.serviceId_entity_extraction.value,
            model: params.model.value,
            inference_api: params.inference_api.value,
        };
        const aiConnector = new InteractiveAIConnector();
        aiConnector.data = inputData;
        aiConnector.scriptParams = {};
        const { data: aiResult } = await aiConnector.process();
        let queryType = (aiResult?.cs_query_type ?? 'unknown').toLowerCase();
        let entities_extractedAi = aiResult?.extracted_entities ?? {};
        console.log(`Classification Prediction: ${queryType}`);
        console.log(entities_extractedAi);
        // entities_extractedAi = this.maskEntities(entities_extractedAi, maskedData)
        entities_extractedAi = Object.entries(entities_extractedAi).filter(([key, val]) => (val != null));
        entities_extractedAi = entities_extractedAi.map(([key, val]) => {
            if (Array.isArray(val) && val.length>0) return [key, maskedData.entity_map[val]];
            if (Object.keys(maskedData.entity_map).includes(val)) return [key, maskedData.entity_map[val][0]]
            else return [key, val]
        });
        entities_extractedAi = Object.fromEntries(entities_extractedAi);
        entities_extractedAi = this.processAiExtractedValue(entities_extractedAi);

        //creating precision object and setting input data into it
        const precisionConnector = new PrecisionccConnector();
        precisionConnector.data = {
            context, rpaParams, entities_extractedAi
        }

        //declare variable to use it outside of scope
        let validEntities = this.hasPrecisionData(entities_extractedAi);
        // await casePage.bringToFront();
        await casePage.waitForSelector('select[data-autoid="cust_272_ctrl"]', { timeout: 90000 });
        const selectedValue = await casePage.$eval(
            'select[data-autoid="cust_272_ctrl"]',
            select => select.value
        );

        if (selectedValue.toLocaleLowerCase() == clientType.toLocaleLowerCase()) {
            try {
                let customerDetailInCrm = await this.case_accountName(casePage);
                if (!customerDetailInCrm) {
                    await this.tagProspect(casePage, from);
                    assignedAgent = await this.assignAgentRoundRobin(this.agents, bucket);

                    await this.editCaseFields(casePage, assignedAgent.empCode);
                    await this.savebuttonClick(casePage);

                    if (queryType.includes('bank') || queryType.includes('demat')) {
                        //login to precisioncc and get data if validEntities is true
                        if (validEntities) {
                            const precisionccdetails = await precisionConnector.process();

                            //check if customer found in precisioncc
                            if (!precisionccdetails || precisionccdetails.rc == 1) {
                                await casePage.bringToFront();
                            }
                            else if (precisionccdetails.rc == 0) {
                                await casePage.bringToFront();
                                precisionData = precisionccdetails.data.details;
                                if (precisionData.dp_Name.toLowerCase().includes("bank")) {
                                    const replyCustomerConnector = new replyToCustomerConnector();
                                    replyCustomerConnector.data = {
                                        casePage, queryType, precisionData, from
                                    }
                                    await replyCustomerConnector.process();
                                    // await this.replyToCustomer(casePage, queryType, precisionData, from);
                                    await this.closeCase(casePage, queryType);
                                    assignedAgent.empCode = "H13351";
                                }
                            }
                        }
                    }

                    await this.setLastCaseId(c.value);

                }
                //  Special-case handling: assign to  agent and exit
                else if (isSpecialCase || !entities_extractedAi) {
                    //  update call type
                    await this.callTypeUpdate(casePage);
                    //also add cusomerTag or prospect tag 
                    assignedAgent = await this.assignAgentRoundRobin(this.agents, bucket);
                    await this.editCaseFields(casePage, assignedAgent.empCode);
                    await this.setTradingNumber(casePage);
                    await this.savebuttonClick(casePage);
                    await this.setLastCaseId(c.value);

                }
                else {
                    //  update call type
                    await this.callTypeUpdate(casePage);
                    //  Assign agent via round robin
                    assignedAgent = await this.assignAgentRoundRobin(this.agents, bucket);
                    //SET tat and assign agent
                    await this.editCaseFields(casePage, assignedAgent.empCode);
                    await this.setTradingNumber(casePage);
                    await this.savebuttonClick(casePage);
                    //  Reply and close based on query type (implement as needed)
                    if (queryType.includes('bank') || queryType.includes('demat')) {
                        //login to precisioncc and get data if validEntities is true
                        if (validEntities) {
                            //precision login and extract value
                            const precisionccdetails = await precisionConnector.process();
                            //check if customer found in precisioncc
                            if (!precisionccdetails || precisionccdetails.rc == 1) {
                                await casePage.bringToFront();
                            }
                            else if (precisionccdetails.rc == 0) {
                                await casePage.bringToFront();
                                precisionData = precisionccdetails.data.details;
                                if (precisionData.dp_Name.toLowerCase().includes("bank")) {
                                    const replyCustomerConnector = new replyToCustomerConnector();
                                    replyCustomerConnector.data = {
                                        casePage, queryType, precisionData, from
                                    }
                                    await replyCustomerConnector.process();
                                    // await this.replyToCustomer(casePage, queryType, precisionData, from);
                                    await this.closeCase(casePage, queryType);
                                    assignedAgent.empCode = "H13351";
                                }
                            }
                        }
                    }
                    await this.setLastCaseId(c.value);
                }
                await casePage.close();
            }
            catch (err) {
                // Log to Excel when error occurs
                await this.getErrorLogAppended(aiResult, queryType, bucket, c, assignedAgent, entities_extractedAi, precisionData, err, casePage, CaseResultFilePath, caseSheetName)
                return { rc: 0 };
            }
        }
        //for prospect--------------------------------------------------------
        else {
            // add prospect tag and assign agent
            try {
                await this.tagProspect(casePage, from);
                assignedAgent = await this.assignAgentRoundRobin(this.agents, bucket);
                await this.editCaseFields(casePage, assignedAgent.empCode);
                await this.savebuttonClick(casePage);
                if (isSpecialCase || !entities_extractedAi) {
                    await this.setLastCaseId(c.value);
                    await casePage.close();
                }
                else if (queryType.includes('bank') || queryType.includes('demat')) {
                    //  Reply and close based on query type (implement as needed)
                    //login to precisioncc and get data if validEntities is true
                    if (validEntities) {

                        const precisionccdetails = await precisionConnector.process();
                        //check if customer found in precisioncc
                        if (!precisionccdetails || precisionccdetails.rc == 1) {
                            await casePage.bringToFront();
                        }
                        else if (precisionccdetails.rc == 0 && precisionccdetails.data != null) {
                            await casePage.bringToFront();
                            precisionData = precisionccdetails.data.details;
                            if (precisionData.dp_Name.toLowerCase().includes("bank")) {
                                const replyCustomerConnector = new replyToCustomerConnector();
                                replyCustomerConnector.data = {
                                    casePage, queryType, precisionData, from
                                }
                                await replyCustomerConnector.process();
                                // await this.replyToCustomer(casePage, queryType, precisionData, from);
                                await this.closeCase(casePage, queryType);
                                assignedAgent.empCode = "H13351";
                            }
                        }
                    }
                    await this.setLastCaseId(c.value);
                    await casePage.close();
                }
                // in case of unknown query or closure or any other case agent is already assigned so close
                else {
                    await this.setLastCaseId(c.value);
                    await casePage.close();
                }
            }
            catch (err) {
                // Log to Excel when error occurs
                await this.getErrorLogAppended(aiResult, queryType, bucket, c, this, assignedAgent, entities_extractedAi, precisionData, err, casePage, CaseResultFilePath, caseSheetName)
                return { rc: 0 };
            }

        }
        return {
            "Buss_id": this.$.buss_id,
            "Bucket Name": bucket,
            "Case Id": c.value,
            "ProcessTime": this.getCurrentFormattedTime(),
            "Classification": queryType,
            "Classification_sentiment": aiResult?.classified_sentiment[0] ?? 'Na',
            "Sentiment": aiResult?.sentiment ?? 'Na',
            "Confidence": aiResult?.confidence ?? 'Na',
            "assignedAgent": assignedAgent?.empCode ?? 'Na',
            "Extracted Values": entities_extractedAi,
            'DpName': precisionData?.dp_Name ?? 'Na',
            "Error": '',
        }
    }
    async getErrorLogAppended(aiResult, queryType, bucket, c, assignedAgent, entities_extractedAi, precisionData, err, casePage, CaseResultFilePath, caseSheetName) {
        const errorResult = {
            "Buss_id": this.$.buss_id,
            "Bucket Name": bucket,
            "Case Id": c.value,
            "ProcessTime": this.getCurrentFormattedTime(),
            "Classification": queryType,
            "Classification_sentiment": aiResult?.classified_sentiment[0] ?? 'Na',
            "Sentiment": aiResult?.sentiment ?? 'Na',
            "Confidence": aiResult?.confidence ?? 'Na',
            "assignedAgent": assignedAgent?.empCode ?? 'Na',
            "Extracted Values": entities_extractedAi,
            'DpName': precisionData?.dp_Name ?? 'Na',
            "Error": err.message,
        }
        await this.appendObjectToExcel(CaseResultFilePath, caseSheetName, errorResult);
        // return errorResult;
        await casePage.close();
    }
    getCurrentFormattedTime() {
        const now = new Date();
        // Get components and pad with '0' if needed
        const month = String(now.getMonth() + 1).padStart(2, '0'); // Months are 0-based
        const day = String(now.getDate()).padStart(2, '0');
        const year = now.getFullYear();
        const hour = String(now.getHours()).padStart(2, '0');
        const min = String(now.getMinutes()).padStart(2, '0');

        // Construct the string
        return `${month}/${day}/${year} ${hour}:${min}`;
    }
    async callTypeUpdate(casePage) {
        // Define a standard timeout for actions within this function
        const actionTimeout = { timeout: 60000 };
        //  Click the Search Icon for Case Category
        await casePage.locator('a[data-autoid="CASE_CATEGORY_srch"]').click(actionTimeout);

        //  Fill "Miscellaneous" into the search box
        await casePage.locator('input[data-autoid="Grid_SearchTextBox_ctrl"]').fill('Miscellaneous', actionTimeout);

        //click button next to miscellaneous
        await casePage.locator('a[data-autoid="gridHF_CASE_CATEGORY"]').click(actionTimeout);

        //  Click the Miscellaneous option in the results
        // await casePage.locator('div[data-autoid="Category_0"]').click(actionTimeout);
        await casePage.locator('text=Miscellaneous').click(actionTimeout);

        // 1. Click the Call Type search icon
        await casePage.locator('a[data-autoid="CASE_SUBCATEGORY_srch"]').click(actionTimeout);

        // 2. Fill 'Demat' in the popup's search box
        await casePage.locator('input[data-autoid="Grid_SearchTextBox_ctrl"]').fill('Demat', actionTimeout);

        // 3. Click the search button in the popup
        await casePage.locator('a[data-autoid="gridHF_CASE_SUBCATEGORY"]').click(actionTimeout);

        // 4. Select 'Demat/Savings' using a text-based locator
        await casePage.locator('text=Demat/Savings').click(actionTimeout);
    }
    // Extract To/CC from the email body
    async extractCustomerEmails(context, crmBaseUrl, caseHref) {
        const casePage = await context.newPage();
        await casePage.goto(`${crmBaseUrl}${caseHref}`, { waitUntil: 'domcontentloaded', timeout: 120000 });
        // 1. Connect to the iframe
        const frameElement = await casePage.waitForSelector('iframe.cke_wysiwyg_frame[title^="Rich Text Editor"]', { timeout: 10000 });
        const frame = await frameElement.contentFrame();
        // 2. Wait for body inside the iframe before evaluate
        await frame.waitForSelector('body', { timeout: 60000 });
        // 2. Extract toEmails, ccEmails, and bodyText
        const { subject, from, toEmails, ccEmails, bodyText } = await frame.evaluate(() => {
            // 1. Try to extract from visible text (works for most cases)
            const bodyText = document.body.innerText;
            const fromIds = bodyText.match(/From:\s*([^\n\r]+)/i);
            const toMatch = bodyText.match(/To:\s*([^\n\r]+)/i);
            const ccMatch = bodyText.match(/CC:\s*([^\n\r]+)/i);
            let subject = bodyText.match(/Subject:\s*([^\n\r]+)/i);

            subject = subject ? subject[1].trim() : "";
            let from = fromIds ? fromIds[1].trim() : " ";
            let toEmails = toMatch ? toMatch[1].split(',').map(e => e.trim()) : [];
            let ccEmails = ccMatch ? ccMatch[1].split(',').map(e => e.trim()) : [];

            // 2. Fallback: If empty, try to extract from mailto: links (covers rare HTML cases)
            if (toEmails.length === 0) {
                toEmails = Array.from(document.querySelectorAll('a[href^="mailto:"]'))
                    .map(a => a.textContent.trim());
            }
            // Remove empty strings and deduplicate
            toEmails = [...new Set(toEmails.filter(Boolean))];
            ccEmails = [...new Set(ccEmails.filter(Boolean))];
            return { subject, from, toEmails, ccEmails, bodyText };

        });
        return { subject, from, toEmails, ccEmails, bodyText, casePage };
    }

    // Edit case fields (TAT Department, Case Owner, Save)
    async editCaseFields(page, agentCode) {
        const actionTimeout = { timeout: 60000 };

        // Set TAT Department to Customer Care
        await page.locator('select[data-autoid="cust_1301_ctrl"]').selectOption('1', actionTimeout);

        // Set Case Owner (assign agent)
        await page.locator('a[data-autoid="CASE_OWNER_srch"]').click(actionTimeout);
        await page.locator('input[data-autoid="Grid_SearchTextBox_ctrl"]').fill(agentCode, actionTimeout);
        await page.locator('a[data-autoid="gridHF_CASE_OWNER"]').click(actionTimeout);
        await page.locator('div[data-autoid="ShortName_0"]').click(actionTimeout);
    }
    async savebuttonClick(page) {
        // Save
        await page.waitForSelector('a[data-autoid="Save"]', { timeout: 40000 });
        await page.click('a[data-autoid="Save"]');
    }
    async tagProspect(page, email_from) {
        //  Select "Prospect" in customer type dropdown
        await page.waitForSelector('select[name="cust_272"]', { timeout: 60000 });
        await page.selectOption('select[name="cust_272"]', 'Prospect');

        // 2. Fill both "from email" input fields
        await page.waitForSelector('input[name="cust_629"]', { timeout: 40000 });
        await page.fill('input[name="cust_629"]', email_from);
        await page.fill('input[name="cust_481"]', email_from);
    }
    async case_accountName(page) {
        let accountName = await page.inputValue('input[name="CASE_ACCOUNTNAME"]');
        if (!accountName || accountName.trim() === '') {
            // console.log('Customer account name is empty');
            return null
            // Add additional form validation/error handling here as needed
        } else {
            console.log('Customer account name: ' + accountName);
            return accountName
        }
    }
    async setTradingNumber(page) {
        // Select Trading Account Number (if required)
        await page.waitForSelector('a[data-autoid="CASE_PRDHOLDINGNUMBER_srch"]', { timeout: 60000 });
        await page.click('a[data-autoid="CASE_PRDHOLDINGNUMBER_srch"]');
        await page.waitForSelector('div[data-autoid^="HoldingNumber_0"]', { timeout: 60000 });
        await page.click('div[data-autoid^="HoldingNumber_0"]');
    }
    async closeCase(page, queryType) {
        await page.waitForSelector('span[data-autoid="tab_1"]', { state: 'visible' });
        await page.locator('span[data-autoid="tab_1"]').click();
        await this.clickEditButton(page, queryType);

        // 1. Switch to the CKEditor iframe (for Detail Description)
        const frameElement = await page.waitForSelector('iframe.cke_wysiwyg_frame[title^="Rich Text Editor"]', { timeout: 60000 });
        const frame = await frameElement.contentFrame();

        // 2. Clear the content and enter the closure text
        // await frame.waitForSelector('body', { timeout: 30000 });
        // await frame.evaluate((text) => {
        //     document.body.innerHTML = text;
        //     document.body.dispatchEvent(new Event('input', { bubbles: true }));
        // }, queryType);

        await frame.evaluate((queryType) => {
            let success = false;
            if (window.parent.CKEDITOR && window.parent.CKEDITOR.instances && window.parent.CKEDITOR.instances["EMAILBODY"]) {
                let editor = window.parent.CKEDITOR.instances["EMAILBODY"];
                editor.setData(queryType);
                editor.updateElement();
                success = true;
                // console.log('CKeditor Instance');
            }
            if (!success) {
                document.body.innerHTML = queryType;
                ['input', 'change', 'keyup'].forEach(e => {
                    document.body.dispatchEvent(new Event(e, { bubbles: true }));
                })
                // console.log('CKeditor event');
            }
        }, queryType);

        // 3. Set Status to "Customer Confirmed"
        await page.waitForSelector('select[data-autoid="CASE_STATUSCODE_ctrl"]', { timeout: 30000 });
        await page.selectOption('select[data-autoid="CASE_STATUSCODE_ctrl"]', { label: 'Customer Confirmed' }); // or value: '5'

        //CLOSE DESCRIPTION
        await page.waitForSelector('input[data-autoid="cust_89_ctrl"]', { timeout: 30000 });
        await page.fill('input[data-autoid="cust_89_ctrl"]', queryType);

        //assign rpa agent while closing the case
        // Set Case Owner (assign agent)
        await page.waitForSelector('a[data-autoid="CASE_OWNER_srch"]', { timeout: 30000 });
        await page.click('a[data-autoid="CASE_OWNER_srch"]');
        await page.waitForSelector('input[data-autoid="Grid_SearchTextBox_ctrl"]', { timeout: 30000 });
        //rpaagent
        await page.fill('input[data-autoid="Grid_SearchTextBox_ctrl"]', "H13351");
        await page.waitForSelector('a[data-autoid="gridHF_CASE_OWNER"]', { timeout: 30000 });
        await page.click('a[data-autoid="gridHF_CASE_OWNER"]');
        await page.waitForSelector('div[data-autoid="ShortName_0"]', { timeout: 30000 });
        await page.click('div[data-autoid="ShortName_0"]');

        // 4. Click the Save button
        await page.waitForSelector('a[data-autoid="Save"]', { timeout: 50000 });
        await page.click('a[data-autoid="Save"]');


    }
    async loginToCRM(page, crmUserId, crmPassword, crmSite) {

        await page.goto(crmSite, { waitUntil: 'domcontentloaded', timeout: 120000 });
        await page.fill("#TxtName", crmUserId);
        await page.fill("#TxtPassword", crmPassword);
        await page.waitForSelector('input[type="submit"][value="Login"]', { timeout: 90000 })

        await page.locator('input[type="submit"][value="Login"]').click({ timeout: 120000 });
    }
    async navigateToBucket(page, bucketId) {
        await util.wait(5000);
        await page.goto("https://crmnxtappuat.hdfcsec.com/UAT/app/CRMNextObject/Home/Case", { waitUntil: 'domcontentloaded', timeout: 180000 });
        await page.waitForSelector('select[data-autoid="QueryCategoryId_ctrl"]', { timeout: 90000 });
        await page.selectOption('select[data-autoid="QueryCategoryId_ctrl"]', '1006');
        await page.selectOption('select[data-autoid="QueryViewId_ctrl"]', bucketId);
        await page.waitForSelector('[data-autoid="gridHF_View0"]', { timeout: 90000 });
        await page.locator('[data-autoid="gridHF_View0"]').click();
        await page.waitForSelector('[data-autoid="gridHF_View0"]', { timeout: 90000 });
        await page.selectOption('select[data-autoid="pagesize_listing"]', '100');
        await page.waitForSelector('a[data-autoid^="CaseID_"]', { state: 'visible' });
        await util.wait(2000);
    }

    async extractCustCareWebCasesFromPage(page) {
        return await page.$$eval('a[data-autoid^="CaseID_"]', (caseLinks) => {
            return caseLinks.map(link => {
                const idDataAutoid = link.getAttribute('data-autoid'); // e.g., "CaseID_2"
                const suffix = idDataAutoid.split('_')[1]; // "2"
                const currentOwnerElement = document.querySelector(`[data-autoid="CurrentOwner_${suffix}"]`);
                if (currentOwnerElement /*&& currentOwnerElement.title === 'Custcareweb'*/) {
                    // Get the CreatedOn value
                    const createdOnElement = document.querySelector(`[data-autoid="CreatedOn_${suffix}"]`);
                    // Defensive null handling
                    const createdOnValue = createdOnElement ? createdOnElement.getAttribute('title') : null;

                    return {
                        value: link.textContent.trim(),
                        href: link.getAttribute('href'),
                        createdOn: createdOnValue // add createdOn property
                    };
                }
                return null;
            }).filter(Boolean);
        });
    }
    async getAllCustcarewebCaseLinks(page, sortBy) {
        let caseData = [];

        while (true) {
            await page.waitForSelector('a[data-autoid^="CaseID_"]', { state: 'visible' });
            const pageRecords = await this.extractCustCareWebCasesFromPage(page);
            caseData.push(...pageRecords);

            const isNextDisabled = await page.locator('a[data-autoid="nextButton_CrmGrid"].disabled').count();
            // isNextDisabled=1;
            if (isNextDisabled > 0) break;

            const firstRowBefore = await page.locator('a[data-autoid^="CaseID_"]').first().getAttribute('href');
            await util.wait(1000);

            const nextButton = page.locator('a[data-autoid="nextButton_CrmGrid"]');
            const isPaginationVisible = await nextButton.isVisible();

            if (!isPaginationVisible) break;

            await Promise.all([
                page.click('a[data-autoid="nextButton_CrmGrid"]'),
                // nextButton.click(),
                page.waitForFunction(
                    (params) => {
                        const el = document.querySelector(params.selector);
                        return el && el.getAttribute('href') !== params.prevHref;
                    },
                    { selector: 'a[data-autoid^="CaseID_"]', prevHref: firstRowBefore },
                    { timeout: 90000 }
                )
            ]);
        }

        // Sort by date only descending (ignoring time)
        caseData.sort((a, b) => {
            // Extract only the date part (DD/MM/YYYY) ignoring time
            const getDateOnly = str => {
                if (!str) return new Date(0); // minimal date for nulls
                // str is like "19/07/2025 6:43 PM"
                // Extract date substring before first space to exclude time
                const datePart = str.split(' ')[0]; // "19/07/2025"
                const [dd, mm, yyyy] = datePart.split('/');
                // Create Date object with time zeroed
                return new Date(`${yyyy}-${mm}-${dd}T00:00:00`);
            };

            const dateA = getDateOnly(a.createdOn);
            const dateB = getDateOnly(b.createdOn);
            if (sortBy.toLowerCase() == 'asc') {
                return dateA - dateB; // ascend order 
            }
            return dateB - dateA; //return desc
        });

        await util.wait(2000);
        return caseData;
    }
    async openCaseAndGetEmail(context, crmBaseUrl, caseHref) {
        const casePage = await context.newPage();
        await casePage.goto(`${crmBaseUrl}${caseHref}`);
        await casePage.waitForSelector('.form-element__control.overflow-auto.maxH10.white-prewrap', { timeout: 60000 });
        const emailBodyText = await casePage.evaluate(() => {
            const el = document.querySelector('.form-element__control.overflow-auto.maxH10.white-prewrap');
            return el ? el.innerText.trim() : '';
        });
        return { casePage, emailBodyText };
    }
    async clickEditButton(page) {
        await page.locator('a[data-autoid="Edit_0"]').click({ timeout: 180000 })
    }
    //get agents from excel
    async loadAgentsFromExcel(filePath, sheetname) {
        let excelApi = new ExcelApi();
        let rslt = await excelApi.readFile(filePath);
        if (rslt.rc != 0) return rslt;
        let agentsInfo = rslt.data[sheetname].dataRows;
        return agentsInfo.slice(1).map(row => {

            return {
                empCode: row[0],
                name: row[1],
                buckets: row[2].split(",").map(b => b.trim().toLowerCase()),
                timeStart: new Date(row[3]),
                timeEnd: new Date(row[4])
            };
        })
    }
    async getMasterEmail(filePath, sheetname) {
        let excelApi = new ExcelApi();
        let rslt = await excelApi.readFile(filePath);
        if (rslt.rc != 0) return rslt;
        let emails = rslt.data[sheetname].dataRows;
        // Skip header rows and non-email values
        let emailIds = emails
            .map(row => row[1])
            .filter(v =>
                typeof v === 'string' &&
                v.includes('@') &&          // Must contain '@'
                !v.toLowerCase().includes("email id") // Skip header text
            );

        return emailIds;
    }

    async getLastAssignedAgent(bucket) {
        const key = 'last_assigned_agent_' + bucket.toLowerCase().replace(/\s+/g, '_');
        const rslt = await markerService.getValue(key);
        if (rslt.rc !== 0) {
            console.warn(`Failed to get last assigned agent for bucket ${bucket}:`, rslt.msg || rslt);
            return null;
        }
        // rslt.data contains the stored empCode or null if not set
        return rslt.data || null;
    }
    async getLastCaseId() {
        const key = 'Last_assigned_case';
        const rslt = await markerService.getValue(key);
        if (rslt.rc !== 0) {
            console.warn(`Failed to get last assigned caseId: `, rslt.msg || rslt);
            return null;
        }
        return rslt.data || null;

    }

    async setLastAssignedAgent(bucket, empCode) {
        const key = 'last_assigned_agent_' + bucket.toLowerCase().replace(/\s+/g, '_');
        const rslt = await markerService.updateValue(key, empCode);
        if (rslt.rc !== 0) {
            console.error(`Failed to update last assigned agent for bucket ${bucket}:`, rslt.msg || rslt);
            throw new Error(`Failed to update last assigned agent for bucket ${bucket}`);
        }
        return rslt;
    }
    async setLastCaseId(caseid) {
        const key = 'Last_assigned_case';
        const rslt = await markerService.updateValue(key, caseid);
        if (rslt.rc !== 0) {
            console.error(`Failed to update last assigned case id ${caseid}: ` + rslt.msg || rslt);
        }
        return rslt;
    }
    async assignAgentRoundRobin(agents, bucket) {
        const availableAgents = this.getAvailableAgents(agents, bucket);
        if (availableAgents.length === 0) throw new Error('No agents available');

        const lastEmpCode = await this.getLastAssignedAgent(bucket);
        let startIndex = 0;

        if (lastEmpCode) {
            const lastIndex = availableAgents.findIndex(a => a.empCode === lastEmpCode);
            startIndex = (lastIndex + 1) % availableAgents.length;
        }

        const assignedAgent = availableAgents[startIndex];

        // Persist assigned agent
        await this.setLastAssignedAgent(bucket, assignedAgent.empCode);

        return assignedAgent;
    }
    getAvailableAgents(agents, bucket, currentTime = new Date()) {
        const bucketLower = bucket.toLowerCase();
        const currentMinutes = this.getMinutesFromMidnight(currentTime);

        return agents.filter(agent => {
            if (!agent.buckets.includes(bucketLower)) return false;

            const startMinutes = this.getMinutesFromMidnight(agent.timeStart);
            const endMinutes = this.getMinutesFromMidnight(agent.timeEnd);

            // Handle overnight shifts (e.g., 10 PM to 6 AM)
            if (endMinutes < startMinutes) {
                // Time range spans midnight
                return currentMinutes >= startMinutes || currentMinutes <= endMinutes;
            } else {
                return currentMinutes >= startMinutes && currentMinutes <= endMinutes;
            }
        });
    }
    getMinutesFromMidnight(date) {
        return date.getHours() * 60 + date.getMinutes();
    }
    async appendObjectToExcel(filePath, sheetName, newObj) {
        let workbook;
        let worksheet;

        //  We must convert them to a Json string.
        for (const key in newObj) {
            if (typeof newObj[key] === 'object' && newObj[key] !== null) {
                newObj[key] = JSON.stringify(newObj[key], null, 2);
            }
        }

        if (fs.existsSync(filePath)) {
            // File exists: Read it
            workbook = XLSX.readFile(filePath);
            worksheet = workbook.Sheets[sheetName];
            if (!worksheet) {
                // Sheet doesn't exist in the existing file, so create it with headers
                worksheet = XLSX.utils.aoa_to_sheet([Object.keys(newObj)]);
                XLSX.utils.book_append_sheet(workbook, worksheet, sheetName);
            }
        } else {
            // File doesn't exist: Create a new workbook and worksheet with headers
            workbook = XLSX.utils.book_new();
            worksheet = XLSX.utils.aoa_to_sheet([Object.keys(newObj)]);
            XLSX.utils.book_append_sheet(workbook, worksheet, sheetName);
        }

        // Convert only the new object's values to an array
        const newRow = Object.values(newObj);

        // Use sheet_add_aoa to append the new row.
        // { origin: -1 } tells xlsx to start appending after the last existing row.
        XLSX.utils.sheet_add_aoa(worksheet, [newRow], { origin: -1 });

        // Write the updated workbook back to the file
        XLSX.writeFile(workbook, filePath);
    }

    hasPrecisionData(extracted_data) {
        let fieldstobeValidated = ["Bank Account Number", "Bank Customer Id", "Client Id", "DP Id", "Demat Account Number", "Login Id", "Trading Account Number"];
        for (let key of Object.keys(extracted_data)) {
            if (fieldstobeValidated.indexOf(key) > -1) {
                if (extracted_data && extracted_data[key] != null && typeof extracted_data[key] != 'undefined') {
                    return true;
                }
            }
        }
        return false;
    }
    getDatedExcelPath(filePath, filename) {
        const now = new Date();
        const yyyy = now.getFullYear();
        const mmm = now.toLocaleString('en-US', { month: 'short' });
        const dd = String(now.getDate()).padStart(2, '0');
        const folderPath = path.join(filePath, yyyy.toString(), mmm, dd);
        fs.mkdirSync(folderPath, { recursive: true });
        return path.join(folderPath, filename);
    }
    processAiExtractedValue(obj) {
        for (const key in obj) {
            if (obj.hasOwnProperty(key)) {
                const value = obj[key];
                if (typeof value === 'string') {
                    // First, remove all spaces from the string
                    const cleanedValue = value.replace(/\s/g, '');

                    // Now, check if the cleaned string starts with "IN" and has a length of 16
                    if (cleanedValue.toLowerCase().startsWith('in') && cleanedValue.length === 16) {
                        const last8Digits = cleanedValue.slice(-8);
                        obj[key] = last8Digits;
                    }
                }
            }
        }
        return obj;
    }


    // maskEmailPII(emailBody) {
    //     const activatePath = "masking_env\\Scripts\\activate.bat";
    //     const pythonScript = "email_entity_masking.py";
    //     const workingDir = "C:\\Bombus\\Actionabl_dev\\Work\\piiMasking";

    //     return new Promise((resolve, reject) => {
    //         const command = `${activatePath} && python ${pythonScript}`;
    //         const child = spawn("cmd.exe", ["/c", command], {
    //             cwd: workingDir,
    //             stdio: ['pipe', 'pipe', 'pipe']
    //         });

    //         let output = '';
    //         let error = '';

    //         child.stdout.on('data', data => output += data.toString());
    //         child.stderr.on('data', data => {
    //             const str = data.toString();
    //             console.error('Python stderr:', str);
    //             error += str;
    //         });

    //         child.on('close', code => {
    //             if (code !== 0) return reject(new Error(error || `Exited with code ${code}`));
    //             try {
    //                 const result = JSON.parse(output.trim());
    //                 resolve(result);
    //             } catch (e) {
    //                 reject(new Error('JSON parse failed: ' + e.message));
    //             }
    //         });

    //         child.stdin.write(emailBody);
    //         child.stdin.end();
    //     });
    // }
    maskEmailPII(emailBody) {
        const pythonPath = "C:\\Bombus\\Actionabl_dev\\Work\\piiMasking\\masking_env\\Scripts\\python.exe";
        const pythonScript = "email_entity_masking.py";
        const workingDir = "C:\\Bombus\\Actionabl_dev\\Work\\piiMasking";

        return new Promise((resolve, reject) => {
            // Run python directly from virtual env
            const child = spawn(pythonPath, [pythonScript], {
                cwd: workingDir,
                stdio: ['pipe', 'pipe', 'pipe']
            });

            let output = '';
            let error = '';

            child.stdout.on('data', data => output += data.toString());

            child.stderr.on('data', data => {
                const str = data.toString();
                console.error('Python stderr:', str);
                error += str;
            });

            child.on('close', code => {
                if (code !== 0) return reject(new Error(error || `Exited with code ${code}`));
                try {
                    const result = JSON.parse(output.trim());
                    resolve(result);
                } catch (e) {
                    reject(new Error('JSON parse failed: ' + e.message));
                }
            });

            child.stdin.write(emailBody);
            child.stdin.end();
        });
    }
    maskEntities(entities_extractedAi, maskedData) {
        if (!entities_extractedAi || !maskedData?.entity_map) return {};

        const filtered = filtered.filter(([key, val]) => {
            if (Array.isArray(val) && val[0] !== null) return true;
            else if (val !== null || val.toString().toLowerCase() !== 'null') return true;
        });

        const mapped = filtered.map(([key, val]) => {
            const mappedVal = maskedData.entity_map[val] ? maskedData.entity_map[val][0] : val;
            return [key, mappedVal]
        })

        return Object.fromEntries(mapped);
    };

    unmaskValues(maskedObj, entityMap) {
        const unmaskedObj = {};
        for (const [key, value] of Object.entries(maskedObj)) {
            if (value === null || value === undefined) {
                // Preserve null or undefined as is
                unmaskedObj[key] = value;
            } else if (Array.isArray(value)) {
                unmaskedObj[key] = value.map(item => (entityMap[item] ? entityMap[item][0] : item));
            } else {
                unmaskedObj[key] = entityMap[value] ? entityMap[value][0] : value;
            }
        }
        return unmaskedObj;
    }

}
module.exports = webRPAScript;
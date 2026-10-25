/**
 * Builds the FinSight survey (user_study/USER_STUDY.md §4) as a Google Form, with a linked
 * response sheet.
 *
 * How to run:
 *   1. Open https://script.google.com → New project.
 *   2. Delete the sample code, paste this whole file, and click Save.
 *   3. Choose createFinSightSurvey in the function menu and click Run.
 *   4. Allow access when Google asks: Advanced → Go to project → Allow. It's your own script.
 *   5. View → Execution log shows the link to share and the edit link.
 *
 * Don't change question wording: scripts/analyse_user_study.py finds columns by their text.
 */
function createFinSightSurvey() {
  const form = FormApp.create('FinSight: Getting Answers from Financial Reports – Survey');
  form.setDescription(
    'We are final-year CSE students at Amrita School of Computing, Coimbatore, building FinSight: ' +
    'a tool that answers questions about a company\'s financial statements (e.g., "How much did total ' +
    'assets grow?") and shows the calculation and the exact rows it used, so you can check the answer ' +
    'yourself. This 5-minute survey helps us design it. Responses are anonymous and used only for ' +
    'academic research. By submitting, you confirm you are 18 or older and agree to this.');
  form.setCollectEmail(false);
  form.setProgressBar(true);
  form.setAllowResponseEdits(false);
  form.setConfirmationMessage('Thank you! Your response helps our final-year project.');

  const agreeScale = '1 = Strongly disagree, 5 = Strongly agree';

  // Section 1: About you
  form.addSectionHeaderItem().setTitle('About you');
  form.addMultipleChoiceItem()
    .setTitle('Which ONE describes you best?')
    .setChoiceValues([
      'Student – Commerce / BBA / MBA / Finance',
      'Student – CA / CMA / CS',
      'Student – Engineering / IT',
      'Working professional – finance / accounts / banking',
      'Working professional – other',
      'Other'])
    .setRequired(true);
  form.addMultipleChoiceItem()
    .setTitle('Do you invest in stocks or mutual funds?')
    .setChoiceValues(['Yes', 'No'])
    .setRequired(true);
  form.addScaleItem()
    .setTitle('How confident are you reading a balance sheet or income statement?')
    .setBounds(1, 5).setLabels('Not at all', 'Very confident')
    .setRequired(true);

  // Section 2: How you do it today
  form.addPageBreakItem().setTitle('How you do it today');
  form.addCheckboxItem()
    .setTitle("When you need a number from a company's financial report (e.g., profit growth), how do you usually get it?")
    .setChoiceValues([
      'I read the report and calculate it myself',
      'Websites or apps (Screener / Moneycontrol / broker app)',
      'I ask ChatGPT or another AI',
      'I ask a friend / teacher / advisor',
      'I have never needed to'])
    .setRequired(true);
  form.addCheckboxItem()
    .setTitle('What is hardest about getting answers from financial reports?')
    .setHelpText('Choose up to 2.')
    .setChoiceValues([
      'Finding the right number in a long report',
      'Knowing which formula to use',
      'Getting the unit right (thousand / lakh / crore / %)',
      'It takes too much time',
      'Not knowing if my answer is correct'])
    .setValidation(FormApp.createCheckboxValidation().setHelpText('Please choose at most 2.').requireSelectAtMost(2).build())
    .setRequired(true);
  form.addMultipleChoiceItem()
    .setTitle('Have you used an AI tool (ChatGPT, Gemini etc.) for questions about financial documents or numbers?')
    .setChoiceValues([
      'Yes – the answers were mostly right',
      'Yes – the answers were sometimes wrong',
      "Yes – but I didn't check them",
      'No'])
    .setRequired(true);

  // Section 3: An example
  form.addPageBreakItem()
    .setTitle('An example')
    .setHelpText(
      "Below is part of a real company's annual report.\n\n" +
      'Total assets by region (in thousands):\n' +
      '• AMER: 2019 = 751,990 | 2018 = 645,791\n' +
      '• APAC: 2019 = 958,744 | 2018 = 937,510\n' +
      '• EMEA: 2019 = 209,541 | 2018 = 193,797');
  form.addMultipleChoiceItem()  // answer key: Wrong (193,797 is the 2018 value)
    .setTitle(`An AI tool was asked "What were EMEA's total assets in 2019?" and answered "193,797 thousand". Is this answer correct?`)
    .setChoiceValues(['Correct', 'Wrong', 'Not sure'])
    .setRequired(true);
  form.addSectionHeaderItem()
    .setTitle('Two tools answer the question "What was the change in APAC total assets between 2018 and 2019?"')
    .setHelpText(
      'TOOL A:  21,234\n\n' +
      'TOOL B:  21,234 thousand\n' +
      'Calculation: 958,744 − 937,510 = 21,234\n' +
      'Source: APAC row, 2019 and 2018 columns (highlighted on the report page)');
  form.addMultipleChoiceItem()
    .setTitle('Which answer would you trust more?')
    .setChoiceValues(['Tool A', 'Tool B', 'No difference'])
    .setRequired(true);
  form.addScaleItem()
    .setTitle("How much would you trust Tool A's answer?")
    .setBounds(1, 5).setLabels('Not at all', 'Completely')
    .setRequired(true);
  form.addScaleItem()
    .setTitle("How much would you trust Tool B's answer?")
    .setBounds(1, 5).setLabels('Not at all', 'Completely')
    .setRequired(true);
  form.addMultipleChoiceItem()
    .setTitle('With Tool B, would you check the answer yourself before using it?')
    .setChoiceValues(['Always', 'Sometimes', "Never – I'd trust it"])
    .setRequired(true);
  form.addMultipleChoiceItem()
    .setTitle('Which part of Tool B is most useful to you?')
    .setChoiceValues(['The calculation', 'The source rows on the page', 'The unit (thousand)',
                      'All are equally useful', 'None'])
    .setRequired(true);

  // Section 4: Your opinion
  form.addPageBreakItem().setTitle('Your opinion');
  form.addGridItem()
    .setTitle('Your opinion')
    .setHelpText(agreeScale)
    .setRows([
      'Seeing the exact rows an answer came from would help me catch mistakes.',
      'Seeing the calculation would help me understand how the number was found.',
      'Showing the unit clearly (thousand / lakh / crore / %) is important to me.',
      'I would rather the tool say "I\'m not sure" than give a possibly wrong answer.',
      'I would prefer a tool that runs on my own computer so my documents stay private.'])
    .setColumns(['1', '2', '3', '4', '5'])
    .setRequired(true);

  // Section 5: What you need
  form.addPageBreakItem().setTitle('What you need');
  form.addCheckboxItem()
    .setTitle('Which features would be most useful to you?')
    .setHelpText('Choose up to 3.')
    .setChoiceValues([
      'Indian number format (lakh/crore and 1,00,000 grouping)',
      'Full annual reports (100+ pages)',
      'Scanned or photographed reports',
      'Comparing two companies or two years',
      'Export answers and sources to Excel or PDF',
      'Questions in Hindi or Tamil or other Indian languages',
      'Explaining financial terms in simple words'])
    .setValidation(FormApp.createCheckboxValidation().setHelpText('Please choose at most 3.').requireSelectAtMost(3).build())
    .setRequired(true);
  form.addMultipleChoiceItem()
    .setTitle('If a tool like FinSight were available, how would you use it?')
    .setChoiceValues(['Regularly', 'Occasionally', 'Only to try it', 'Not at all'])
    .setRequired(true);
  form.addParagraphTextItem()
    .setTitle('Anything you would want such a tool to do?')
    .setRequired(false);

  // Responses go to a Google Sheet (File → Download → CSV for the analysis script).
  const sheet = SpreadsheetApp.create('FinSight survey responses');
  form.setDestination(FormApp.DestinationType.SPREADSHEET, sheet.getId());

  Logger.log('SHARE THIS LINK:  ' + form.shortenFormUrl(form.getPublishedUrl()));
  Logger.log('Edit the form:    ' + form.getEditUrl());
  Logger.log('Responses sheet:  ' + sheet.getUrl());
}

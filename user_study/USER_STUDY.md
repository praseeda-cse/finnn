# FinSight User Study: use case, users, questionnaire, analysis

Technical Enrichment Activity #7 (User Study / Field Study). Required: study plan, questionnaire,
feedback from ≥30 participants, analysis, improvements incorporated.
**Get the guide's approval for this plan before sending the form.**

Type of study: a **requirements and perception survey**. Participants do not use the system. They
answer questions about how they handle financial reports today, judge a written example of FinSight's
output, and say what they need. Everything fits in a Google Form; no demo or screenshots are needed.

---

## 1. Final use case

> **Ask a question about a company's financial statement and get the answer, with its calculation,
> its unit and the exact rows it came from highlighted on the page, so you can check it in seconds
> instead of searching and calculating by hand.**

The difference from ChatGPT: FinSight shows **where** each number came from and **computes** the arithmetic exactly.

## 2. Who the users are

| Group | Who | What they need |
|---|---|---|
| **Learners** | Commerce / BBA / MBA / finance students, CA / CMA / CS students | the formula and working, to learn and to check their own answers |
| **Investors** | anyone who invests in stocks or mutual funds (Q2) | plain answers with units they understand (lakh/crore) |
| **Professionals** | audit, accounts, analysis, banking | traceability, accuracy, privacy |
| **Technical** | engineering / IT students | general opinion (not target users) |

**Recruitment (40 invited → ≥30 valid).** Try to keep engineering/IT friends to at most half of the responses.
For the rest, ask friends in commerce or CA, and family members who invest or work in banks or accounts.

---

## 3. Study plan

**Research questions**
- RQ1 Problem: how do people get numbers from financial reports today, and what is hard about it?
- RQ2 Verification: do people notice a wrong AI answer when no source is shown?
- RQ3 Trust: do people trust an answer more when it shows the calculation and the source rows?
- RQ4 Needs: which features matter to each group, and would they use such a tool?

**Procedure (~5 min).** Online Google Form, shared on WhatsApp or by e-mail. Anonymous.

**Ethics.** Anonymous, no names or emails, voluntary, used only for the project report and papers.

**Timeline.** Pilot with 3 friends → fix anything confusing → collect for about a week → analyse →
build the top requested improvements into FinSight → write up.

---

## 4. Questionnaire (19 questions)

**Form-building rules** (so `scripts/analyse_user_study.py` can read the export):
- Keep the question wording exactly as written.
- In the grid (Q13–17), label the columns **1 2 3 4 5**, not words.
- Settings → "Collect email addresses" OFF.
- Every question is Required except Q19.

**Title:** FinSight: Getting Answers from Financial Reports – Survey

**Description:**
> We are final-year CSE students at Amrita School of Computing, Coimbatore, building FinSight: a tool
> that answers questions about a company's financial statements (e.g., "How much did total assets
> grow?") and shows the calculation and the exact rows it used, so you can check the answer yourself.
> This 5-minute survey helps us design it. Responses are anonymous and used only for academic research.
> By submitting, you confirm you are 18 or older and agree to this.

### Section 1: About you
1. **Which ONE describes you best?** *(Multiple choice)*
   - Student – Commerce / BBA / MBA / Finance
   - Student – CA / CMA / CS
   - Student – Engineering / IT
   - Working professional – finance / accounts / banking
   - Working professional – other
   - Other
2. **Do you invest in stocks or mutual funds?** *(Yes · No)*
3. **How confident are you reading a balance sheet or income statement?** *(Linear scale 1–5: Not at all → Very confident)*

### Section 2: How you do it today
4. **When you need a number from a company's financial report (e.g., profit growth), how do you usually get it?** *(Checkboxes)*
   - I read the report and calculate it myself
   - Websites or apps (Screener / Moneycontrol / broker app)
   - I ask ChatGPT or another AI
   - I ask a friend / teacher / advisor
   - I have never needed to
5. **What is hardest about getting answers from financial reports?** *(Checkboxes; Response validation: select at most 2)*
   - Finding the right number in a long report
   - Knowing which formula to use
   - Getting the unit right (thousand / lakh / crore / %)
   - It takes too much time
   - Not knowing if my answer is correct
6. **Have you used an AI tool (ChatGPT, Gemini etc.) for questions about financial documents or numbers?** *(Multiple choice)*
   - Yes – the answers were mostly right
   - Yes – the answers were sometimes wrong
   - Yes – but I didn't check them
   - No

### Section 3: An example
**Section description (paste exactly):**
> Below is part of a real company's annual report.
>
> Total assets by region (in thousands):
> • AMER: 2019 = 751,990 | 2018 = 645,791
> • APAC: 2019 = 958,744 | 2018 = 937,510
> • EMEA: 2019 = 209,541 | 2018 = 193,797

7. **An AI tool was asked "What were EMEA's total assets in 2019?" and answered "193,797 thousand". Is this answer correct?** *(Correct · Wrong · Not sure)*
   *(Answer key: **Wrong**. 193,797 is the 2018 value; the 2019 value is 209,541. This is a "wrong column" error, the kind Paper 1 found to be common.)*

**Text block before Q8 (Google Forms: "Add title and description"):**
> Two tools answer the question "What was the change in APAC total assets between 2018 and 2019?"
>
> **Tool A:** 21,234
>
> **Tool B:** 21,234 thousand
> Calculation: 958,744 − 937,510 = 21,234
> Source: APAC row, 2019 and 2018 columns (highlighted on the report page)

8. **Which answer would you trust more?** *(Tool A · Tool B · No difference)*
9. **How much would you trust Tool A's answer?** *(Linear scale 1–5: Not at all → Completely)*
10. **How much would you trust Tool B's answer?** *(Linear scale 1–5: Not at all → Completely)*
11. **With Tool B, would you check the answer yourself before using it?** *(Always · Sometimes · Never – I'd trust it)*
12. **Which part of Tool B is most useful to you?** *(Multiple choice)*
   - The calculation
   - The source rows on the page
   - The unit (thousand)
   - All are equally useful
   - None

### Section 4: Your opinion
**Q13–17 are one multiple-choice grid** titled "Your opinion". Columns: 1 2 3 4 5 (1 = Strongly disagree, 5 = Strongly agree). Rows:

13. Seeing the exact rows an answer came from would help me catch mistakes.
14. Seeing the calculation would help me understand how the number was found.
15. Showing the unit clearly (thousand / lakh / crore / %) is important to me.
16. I would rather the tool say "I'm not sure" than give a possibly wrong answer.
17. I would prefer a tool that runs on my own computer so my documents stay private.

### Section 5: What you need
18. **Which features would be most useful to you?** *(Checkboxes; Response validation: select at most 3)*
   - Indian number format (lakh/crore and 1,00,000 grouping)
   - Full annual reports (100+ pages)
   - Scanned or photographed reports
   - Comparing two companies or two years
   - Export answers and sources to Excel or PDF
   - Questions in Hindi or Tamil or other Indian languages
   - Explaining financial terms in simple words
19. **If a tool like FinSight were available, how would you use it?** *(Multiple choice)*
   - Regularly
   - Occasionally
   - Only to try it
   - Not at all

20. *(optional, not counted)* **Anything you would want such a tool to do?** *(Paragraph)*

**Why each question is there**

| Questions | What they give you |
|---|---|
| 1–3 | groups for comparison, plus financial background |
| 4–6 | the problem today (motivation for the project) |
| 7 | **key result:** how many people accept a wrong AI answer when no source is shown |
| 8–12 | whether showing the calculation and source increases trust (a paired test), and which part matters |
| 13–17 | support for each design choice: evidence, calculation, units, abstaining, running offline |
| 18–19 | what to build next, and whether people would use it |

---

## 5. Analysis plan

Export the responses to CSV, then run `python scripts/analyse_user_study.py responses.csv`.
Results are reported overall and per group.

| What | How |
|---|---|
| Who took part | counts per group (Q1) and investors (Q2) |
| Problem today | counts for Q4–Q6 |
| Spotting a wrong answer | % who said **Wrong** on Q7, by group and by confidence (Q3) |
| A vs. B trust | Q9 vs. Q10: Wilcoxon signed-rank test; % choosing B (Q8) |
| Would still verify | Q11 counts |
| Design choices | % agree (4–5) for Q13–17, by group |
| Features, adoption | counts for Q18 and Q19, by group |
| Open answers (Q20) | group into themes and count them (done by hand) |

## 6. Improvements loop (needed for marks)

Make a table: *what users asked for → how many → change made in FinSight → where in the code*.
Likely changes:
- Indian number format (₹, lakh/crore, 1,00,000 grouping) and the scale written out ("21,234 thousand = 2.12 crore").
- Formula in words next to the equation ("change = 2019 − 2018").
- An "I'm not sure" message when the votes disagree or an operand isn't on the page (Paper 2 already has both signals).
- Export the evidence and equation to CSV.

Show each change with a before/after screenshot of the demo output.

## 7. Evidence to submit
- Guide approval
- This plan
- Form link and a PDF print of the form
- Anonymised response CSV
- Script output and charts
- Improvements table with before/after screenshots

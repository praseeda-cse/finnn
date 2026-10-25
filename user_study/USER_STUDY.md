# FinSight User Study: use case, users, questionnaire, analysis

Technical Enrichment Activity #7 (User Study / Field Study). Required: study plan, questionnaire,
feedback from ≥30 participants, analysis, improvements incorporated.
**Get the guide's approval for this plan before sending the form.**

---

## 1. Final use case

> **Ask a question about a company's financial statement and get the answer, with its calculation,
> its unit and the exact rows it came from highlighted on the page, so you can check it in seconds
> instead of searching and calculating by hand.**

Scope today: digital PDFs of 1–3 pages (a balance sheet, P&L, cash-flow or segment table).
Full annual reports, scanned pages and Indian formats are *asked about* as future features, not promised.
The difference from ChatGPT: FinSight shows **where** each number came from and **computes** the arithmetic exactly.

## 2. Who the users are

| Group | Who | What they need from FinSight |
|---|---|---|
| **Learners** | Commerce / BBA / MBA / finance students, CA / CMA / CS students | the formula and working, to learn and to check their own answers |
| **Investors** | anyone who invests in stocks or mutual funds (Q2) | plain answers with units they understand (lakh/crore) |
| **Professionals** | audit, accounts, analysis, banking | traceability, accuracy, privacy (runs offline), export |
| **Technical** | engineering / IT students | usability feedback only (not target users) |

**Recruitment (40 invited → ≥30 valid).** Try to keep engineering/IT friends to at most half of the responses.
For the rest, ask friends in commerce or CA, and family members who invest or work in banks or accounts.

---

## 3. Study plan

**Research questions**
- RQ1 Usability: is the demo easy to use? (UMUX-Lite, converted to a SUS-equivalent 0–100 score)
- RQ2 Verification: do the equation and highlighted evidence help users check an answer and *catch a wrong one*?
- RQ3 Trust: do users trust an explained answer more than a bare one?
- RQ4 Fit: which features matter for each group, and would they use it?

**Procedure (~7 min).** Profile → two tasks in the demo → bare vs. explained answer → opinion grid →
features → one open question. If possible, run it with each friend in person or on a call: you start
the demo and they type the questions.

**Materials**
- Demo link (notebook 03, Gradio) that stays up during collection, or run it live.
- Task page: the APAC total-assets segment table (Paper 3, Fig. 2). **Task B** must be a question
  where FinSight is **wrong**: pick one from the test run where the prediction ≠ gold.
- Two screenshots: **A** = only "Answer: 21,234"; **B** = answer + unit (thousand) + equation
  `958,744 − 937,510` + highlighted APAC row + confidence.

**Ethics.** Anonymous, no names or emails, voluntary, used only for the project report and papers.

**Timeline.** Pilot with 3 friends → fix anything confusing → collect for about a week → analyse →
build the top improvements → re-test with 5–10 people → write up.

---

## 4. Questionnaire (19 questions)

**Form-building rules** (so `scripts/analyse_user_study.py` can read the export):
- Keep the question wording exactly as written.
- In the grid (Q12–16), label the columns **1 2 3 4 5**, not words.
- Settings → "Collect email addresses" OFF.
- Every question is Required except Q19.

**Title:** FinSight: Checking Answers from Financial Statements – User Study

**Description:**
> We are final-year CSE students at Amrita School of Computing, Coimbatore. FinSight answers
> questions about a company's financial statements and shows the calculation and the exact rows on
> the page it used, so you can check the answer yourself. Takes about 7 minutes. Responses are
> anonymous and used only for academic research. By submitting, you confirm you are 18 or older
> and agree to this.

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
4. **Have you used an AI tool (ChatGPT, Gemini etc.) for questions about financial documents or numbers?** *(Multiple choice)*
   - Yes – the answers were mostly right
   - Yes – the answers were sometimes wrong
   - Yes – but I didn't check them
   - No

### Section 2: Try FinSight
Description: *Open the demo: [LINK]. The report page is already loaded. Ask each question exactly as
written, look at the answer and the highlighted page, then answer below.*

**Task A – ask: "What was the change in the total assets from APAC between 2018 and 2019?"**

5. **Task A: Is FinSight's answer correct? Check it against the page.** *(Correct · Wrong · Not sure)*

**Task B – ask: "[QUESTION WHERE FINSIGHT IS WRONG]"**

6. **Task B: Is FinSight's answer correct? Check it against the page.** *(Correct · Wrong · Not sure)*
7. **For Task B, what did you use to decide?** *(Checkboxes)*
   - The highlighted rows on the page
   - The equation
   - The unit
   - My own reading of the page
   - I guessed
8. **Without FinSight, how long would these two questions take you by hand?** *(Multiple choice)*
   - Less than 2 min
   - 2–5 min
   - 5–10 min
   - More than 10 min
   - I couldn't do it

### Section 3: Answer only vs. explained answer
Description: *Both screenshots answer the same question.* [Screenshot A] [Screenshot B]

9. **Which answer do you trust more?** *(A – answer only · B – answer with equation and evidence · No difference)*
10. **How much do you trust Answer A?** *(Linear scale 1–5: Not at all → Completely)*
11. **How much do you trust Answer B?** *(Linear scale 1–5: Not at all → Completely)*

### Section 4: Your opinion
**Q12–16 are one multiple-choice grid** titled "Your opinion". Columns: 1 2 3 4 5 (1 = Strongly disagree, 5 = Strongly agree). Rows:

12. FinSight's capabilities meet my requirements.
13. FinSight is easy to use.
14. The highlighted evidence made it easy to check where the answer came from.
15. The equation helped me understand how the number was calculated.
16. I would rather the system say "I'm not sure" than give a possibly wrong answer.

### Section 5: Features and feedback
17. **Which features would be most useful to you?** *(Checkboxes, Response validation: select at most 3)*
   - Indian number format (lakh/crore and 1,00,000 grouping)
   - Full annual reports (100+ pages)
   - Scanned or photographed reports
   - Comparing two companies or two years
   - Export answers and evidence to Excel or PDF
   - Runs offline on my own computer (documents stay private)
   - Questions in Hindi or Tamil or other Indian languages
18. **If FinSight were a real product, how would you use it?** *(Regularly · Occasionally · Only to try it · Not at all)*
19. **What one change would make FinSight more useful for you?** *(Paragraph, optional)*

**Why each question is there**

| Questions | What they give you |
|---|---|
| 1–4 | groups for comparison (learner / investor / professional / technical) and background |
| 5–7 | **the key result:** can people catch a wrong answer, and does the evidence help them do it? |
| 8 | time saved, which supports the use case |
| 9–11 | do explanations increase trust? (paired test) |
| 12–13 | UMUX-Lite, a standard 2-item usability scale (Lewis et al., 2013) |
| 14–16 | whether each explanation feature helps; support for abstaining with "I'm not sure" |
| 17–18 | what to build next, and whether people would actually use it |
| 19 | themes for the improvements table |

---

## 5. Analysis plan

Export the responses to CSV, then run `python scripts/analyse_user_study.py responses.csv`.
Results are reported overall and per group.

| What | How |
|---|---|
| Who took part | counts per group (Q1) and investors (Q2) |
| Catching a wrong answer | % who said **Wrong** on Task B; compare participants who used evidence or the equation (Q7) with those who didn't |
| Task A | % who judged it correctly |
| Usability | UMUX-Lite = ((Q12 − 1) + (Q13 − 1)) / 8 × 100; SUS-equivalent = 0.65 × UMUX-Lite + 22.9 (Lewis et al., 2013); 68 = average |
| A vs. B trust | Q10 vs. Q11: Wilcoxon signed-rank; % choosing B |
| Explanation features | % agree (4–5) for Q14–16, per group |
| Features, adoption | counts for Q17 and Q18, per group |
| Open answers | group Q19 into themes and count them (done by hand) |

## 6. Improvements loop (needed for marks)

Make a table: *feedback theme → how many people raised it → change made → where in the code or demo*.
Changes the team can realistically build:
- Indian format toggle (₹, lakh/crore, 1,00,000 grouping) and the scale written out ("21,234 thousand = 2.12 crore").
- Formula in words next to the equation ("change = 2019 − 2018").
- An "I'm not sure" message when the votes disagree or an operand isn't on the page (Paper 2 already has both signals).
- Export the evidence and equation to CSV.

Re-test the changes with 5–10 people and add a before/after row.

## 7. Evidence to submit
- Guide approval
- This plan
- Form link and a PDF print of the form
- Anonymised response CSV
- Script output and charts
- Improvements table with before/after screenshots

# FinSight User Study: use case, users, questionnaire, analysis

Technical Enrichment Activity #7 (User Study / Field Study). Required: study plan, questionnaire,
feedback from ≥30 participants, analysis, improvements incorporated.
**Get the guide's approval for this plan before sending the form.**

---

## 1. Final use case

> **Ask a question about a company's financial statement and get the answer, with its calculation,
> its unit and the exact rows it came from highlighted on the page, so you can check it in seconds
> instead of searching and calculating by hand.**

Scope today: digital PDFs of 1–3 pages (a balance sheet, P&L, cash-flow or segment table, or a page
of notes). Full 200-page reports, scanned pages and Indian formats are *asked about* in the survey
as future features, not promised.

Why this use case: it is what the system really does (Paper 2), and it puts the system's main
difference from ChatGPT up front. FinSight shows **where** each number came from and **computes**
the arithmetic exactly instead of guessing it.

## 2. Who the users are

| Persona | Who | Their job with a financial statement | What they need from FinSight | Hypothesis we test |
|---|---|---|---|---|
| **L: Learner** | B.Com, BBA, MBA, M.Com and finance students; CA/CMA/CS students | ratio analysis and case-study assignments, exam practice | the **formula and working**, to learn and to check their own answers | the equation view helps them understand (F2 higher for L) |
| **I: Retail investor** | people who invest in stocks or mutual funds | judge a company before buying: growth, debt, margins | plain answers with **units they understand** (lakh/crore), no jargon | they skip annual reports because they're too long or technical; units matter (F3) |
| **P: Finance professional** | auditors and article assistants, accountants, analysts, bank credit officers | extract and cross-check numbers across many statements, quickly and correctly | **traceability** (an audit trail), accuracy, **privacy** of client documents, export | they always verify; evidence highlighting is the feature they value most (F1); privacy blocks online tools |
| **T: Technical evaluator** | engineering / IT students and professionals | not a target user | usability feedback only | (no persona hypothesis; SUS only) |

**Recruitment quotas (40 invited → ≥30 valid):** L ≥ 12 · I ≥ 8 · P ≥ 6 · T ≤ 8.
Classmates are easy to recruit, so cap **T** or the results will mostly reflect CSE students. Good places to find each group:
- L: commerce and management departments, CA coaching groups.
- I: family members and seniors who invest.
- P: relatives in banks or CA firms, and CA article assistants. Try to get at least 6 even if it's slow.

---

## 3. Study plan

**Research questions**
- RQ1 Usability: can each persona use the demo easily? (System Usability Scale, SUS)
- RQ2 Verification: do the equation and the highlighted evidence help users check an answer, and *catch a wrong one*?
- RQ3 Trust: do users trust an explained answer more than a bare one? Do they want "I'm not sure" instead of a guess?
- RQ4 Fit: does the use case match each persona's real work, and what is missing (Indian formats, long reports, privacy, export)?

**Procedure (~12 min).** Consent → profile → **persona section** (the form branches on Q2) → three
tasks in the demo → bare vs. explained answer → SUS → trust → features → open feedback.
Run sessions in person or on a call where possible: you start the demo and the participant types the questions.
This keeps the demo stable and lets you note what confuses people.

**Materials**
- Demo link (notebook 03, Gradio) that stays up during collection, or run it live.
- Task page: the APAC total-assets segment table (Paper 3, Fig. 2). Task 3 must be a question
  where FinSight is **wrong**: pick one from the test run where the prediction ≠ gold on a similar page.
- Two screenshots for Section D: **A** = only "Answer: 21,234"; **B** = answer + unit (thousand)
  + equation `958,744 − 937,510` + highlighted APAC row + confidence.
- A private answer key (not in the form).

**Ethics.** Anonymous, no names or emails, voluntary, used only for the project report and papers.

**Timeline.** Pilot with 3 people (one each from L, I, P) → fix confusing questions → collect for 7–10 days →
analyse → build the top improvements → re-test them with 5–10 people → write up.

---

## 4. Questionnaire (Google Forms)

**Form-building rules** (so `scripts/analyse_user_study.py` can read the export):
- Keep the question wording exactly as written.
- Grids (Sections E and F) use columns labelled **1 2 3 4 5**, not words.
- Don't put ", " inside checkbox options.
- Settings → "Collect email addresses" OFF.
- Mark every question Required except those marked *(optional)*.
- **Branching:** on Q2, use "Go to section based on answer": L options → Section L, investor → Section I,
  professional → Section P, engineering/other → Section C. At the end of Sections L, I and P, choose "Continue to Section C".

Form title: **FinSight: Checking Answers from Financial Statements – User Study**

Form description:
> We are final-year CSE students at Amrita School of Computing, Coimbatore. FinSight answers
> questions about a company's financial statements, such as "What was the change in total assets?",
> and shows the calculation and the exact rows on the page it used, so you can check the answer
> yourself. This survey takes about 12 minutes. Responses are anonymous and used only for academic research.

### Section A: Consent
1. I am 18 or older and agree that my anonymous responses may be used for academic research. *(Checkbox: "I agree")*

### Section B: About you
2. Which ONE describes you best? If more than one fits, choose the one you would use FinSight for most. *(Multiple choice, with branching)*
   - Student – Commerce / BBA / MBA / Finance → **L**
   - Student – CA / CMA / CS (including articleship) → **L**
   - Retail investor (I invest in stocks or mutual funds) → **I**
   - Finance professional (audit / accounts / analysis / banking) → **P**
   - Student or professional – Engineering / IT → **C**
   - Other → **C**
3. How often do you read financial statements or annual reports? *(Never · A few times a year · Monthly · Weekly · Daily)*
4. How confident are you reading a balance sheet or income statement? *(Linear scale 1–5: Not at all → Very confident)*
5. How often do you use AI tools (ChatGPT, Gemini, Copilot, etc.)? *(Never · Rarely · Weekly · Daily)*
6. Have you ever asked an AI tool a question about a financial document or numbers? *(Yes / No)*
7. If yes, was the answer correct? *(optional; Always correct · Mostly correct · Sometimes wrong · Often wrong · I didn't check)*

### Section L: Learners (students)
L1. How often do your courses or exam preparation require calculating values from financial statements (ratios, growth, changes)? *(Never · Few times a semester · Monthly · Weekly)*
L2. What is hardest when you do this? *(Checkboxes, max 2)*
   - Finding the right number on the page · Knowing which formula to use · Getting the unit right (thousand / lakh / crore / %) ·
     Long reports with many tables · Checking whether my answer is correct
L3. How do you check your answers now? *(Checkboxes)*
   - Solutions or the textbook · Friends · Teacher · ChatGPT or another AI · I usually don't check
L4. Seeing the formula and the source rows would help me *learn* the method, not just get the answer. *(1–5: Strongly disagree → Strongly agree)*
L5. I would use a tool like this to check my own working after solving a problem. *(1–5)*
L6. Using a tool like this might make students depend on it instead of learning. *(1–5)*

### Section I: Retail investors
I1. How long have you been investing? *(< 1 year · 1–3 years · 3–5 years · > 5 years)*
I2. Do you read a company's annual report or financial statements before investing? *(Always · Sometimes · Rarely · Never)*
I3. If you don't always read them, why not? *(Checkboxes; optional)*
   - Too long · Too technical · I don't know where to look · I don't have time · Apps and websites already show the numbers
I4. Where do you get company numbers now? *(Checkboxes)*
   - Screener / Tickertape / Moneycontrol · Broker app (Zerodha / Groww etc.) · Annual report or exchange filings ·
     YouTube or news · ChatGPT or another AI · Friends or advisors
I5. Which questions would you most want to ask about a company? *(Checkboxes, max 3)*
   - Revenue or profit growth · Profit margin · Debt and borrowings · Cash flow · Dividends · Segment or region performance · Expenses breakdown
I6. I would use an AI answer for an investment decision only if I could see where the number came from. *(1–5)*

### Section P: Finance professionals
P1. Your role *(Auditor / article assistant · Accountant · Financial or equity analyst · Bank credit or loan officer · Other)*
P2. Hours per week you spend finding, copying or cross-checking numbers from financial statements *(< 2 · 2–5 · 5–10 · > 10)*
P3. Tools you use for this now *(Checkboxes: Manual reading + Excel · Paid databases (Prowess / Capitaline / Bloomberg) · ChatGPT / Copilot · Company software · Other)*
P4. How serious is the cost of a wrong number in your work? *(1–5: Minor → Very serious)*
P5. Could you upload client or company documents to an online AI tool? *(Yes · Only if it runs on our own systems · No – documents are confidential · Not sure)*
P6. What must an AI tool have before you would use it at work? *(Checkboxes, max 3)*
   - Shows the source of every number (audit trail) · Accuracy above 95% · Runs offline / on-premise ·
     Works on scanned documents · Handles full 200+ page reports · Exports to Excel · Says when it is unsure
P7. Highlighting the exact source rows would save me time when reviewing numbers. *(1–5)*

### Section C: Try FinSight (all participants)
Section description: *Open the demo here: [LINK]. The report page is already loaded. Ask each question
below exactly as written, look at the answer and the highlighted page, then answer.*

**Task 1 – "What was the total assets from AMER in 2018?"** (lookup)
8. What answer did FinSight give? *(Short answer)*
9. Is the answer correct? Check it against the page. *(Correct / Wrong / Not sure)*
10. Did the highlighted row help you check it? *(1–5: Not at all → A lot)*

**Task 2 – "What was the change in the total assets from APAC between 2018 and 2019?"** (calculation)
11. What answer did FinSight give, including the unit? *(Short answer)*
12. Is the answer correct? *(Correct / Wrong / Not sure)*
13. Did seeing the equation (e.g., 958,744 − 937,510) help you check it? *(1–5: Not at all → A lot)*

**Task 3 – [question where FinSight is wrong; fill in]**
14. What answer did FinSight give? *(Short answer)*
15. Is the answer correct? *(Correct / Wrong / Not sure)*
16. What did you use to decide? *(Checkboxes)*
   - The highlighted rows on the page · The equation · The unit · The confidence score · My own reading of the page · I guessed
17. How long did the three tasks take in total? *(< 2 min · 2–5 min · 5–10 min · > 10 min)*
18. How long would the same three questions take you by hand, without FinSight? *(< 2 min · 2–5 min · 5–10 min · > 10 min · I couldn't do it)*

### Section D: Bare answer vs. explained answer
Section description: *Both screenshots answer the same question. [Screenshot A] [Screenshot B]*

19. Which answer do you trust more? *(A – answer only · B – answer with equation and evidence · No difference)*
20. How much do you trust Answer A? *(1–5: Not at all → Completely)*
21. How much do you trust Answer B? *(1–5: Not at all → Completely)*
22. In B, which parts are useful? *(Checkboxes: Highlighted evidence rows · Equation · Unit / scale (thousand / million / %) · Confidence score · None)*

### Section E: Usability (System Usability Scale)
*This is a standard scale, so keep the wording exactly. Grid title "Usability", columns 1–5 (Strongly disagree → Strongly agree).*

23. I think that I would like to use this system frequently.
24. I found the system unnecessarily complex.
25. I thought the system was easy to use.
26. I think that I would need the support of a technical person to be able to use this system.
27. I found the various functions in this system were well integrated.
28. I thought there was too much inconsistency in this system.
29. I would imagine that most people would learn to use this system very quickly.
30. I found the system very cumbersome to use.
31. I felt very confident using the system.
32. I needed to learn a lot of things before I could get going with this system.

### Section F: Trust and explanations
*Grid title "Trust", columns 1–5 (Strongly disagree → Strongly agree).*

33. The highlighted evidence made it easy to check where the answer came from.
34. The equation helped me understand how the number was calculated.
35. Showing the unit (thousand / million / %) is important to me.
36. I would rather the system say "I'm not sure" than give a possibly wrong answer.
37. I would check the page myself before using an AI answer in real work.
38. To show you are reading carefully, please select 4 for this row. *(attention check)*
39. I would trust FinSight more than a general chatbot (e.g., ChatGPT) for questions on financial reports.

### Section G: Fit and features
40. Which features would be most useful to you? *(Checkboxes, max 3)*
   - Indian number format (lakh/crore and 1,00,000 grouping) · Scanned or photographed reports ·
     Full annual reports (100+ pages) · Questions in Hindi or Tamil or other Indian languages ·
     Comparing two companies or two years · Export answers and evidence to Excel or PDF ·
     Runs offline on my own computer · Follow-up questions in a chat
41. If FinSight were a real product, how would you use it? *(Regularly · Occasionally · Only to try it · Not at all)*
42. How likely are you to recommend FinSight to a friend or colleague? *(Linear scale 0–10)*

### Section H: Open feedback
43. What was confusing or frustrating? *(Paragraph)*
44. What one change would make FinSight more useful for you? *(Paragraph)*
45. Any other comments? *(Paragraph, optional)*

---

## 5. Analysis plan

Export responses → CSV → `python scripts/analyse_user_study.py responses.csv`. The script drops
duplicates and failed attention checks, and reports everything below **overall and per persona**.

| What | How | Answers |
|---|---|---|
| Who took part | counts per persona (Q2) and profile (Q3–Q7) | quotas met? |
| Persona answers | counts for L1–L6, I1–I6 and P1–P7 | RQ4: does the use case match their work? |
| Task accuracy | % "Correct" on Tasks 1–2; % who said **Wrong** on Task 3 | RQ2 |
| Catching errors | Task 3 detection rate: participants who used evidence or the equation vs. those who didn't | RQ2 |
| Time saved | Q17 vs. Q18 | value of the use case |
| SUS | odd items (score − 1), even items (5 − score), sum × 2.5 → 0–100; mean ± SD; 68 = average | RQ1 |
| A vs. B trust | Q20 vs. Q21: Wilcoxon signed-rank; % choosing B | RQ3 |
| Trust items | % agree (4–5) per item, per persona | RQ3 and persona hypotheses |
| Personas differ? | SUS and trust in B across L / I / P / T: Kruskal–Wallis | does background matter |
| Features, adoption, NPS | counts for Q40, Q41; NPS = % scoring 9–10 − % scoring 0–6 | RQ4 |
| Open answers | group into themes and count each (done by hand) | improvements |

## 6. Improvements loop (needed for marks)

Make a table: *feedback theme → how many people raised it → change made → where in the code or demo*.
Changes the team can realistically build:
- **Learners:** show the formula in words ("change = 2019 − 2018") next to the equation.
- **Investors:** an Indian format toggle (₹, lakh/crore, 1,00,000 grouping), and the scale written out
  ("21,234 thousand = 2.12 crore").
- **Professionals:** "Export evidence + equation to Excel/CSV"; stress that the model runs locally
  (open TAT-LLM weights), so documents never leave the machine.
- **Everyone:** an "I'm not sure" message when the votes disagree or an operand isn't found on the page
  (Paper 2 already has both signals).

Re-test with 5–10 people and add a before/after row. That's the strongest evidence you can submit.

## 7. Evidence to submit
- Guide approval
- This plan
- Form link and a PDF print of the form
- Anonymised response CSV
- Script output and charts
- Improvements table with before/after screenshots

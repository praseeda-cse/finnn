# FinSight User Study: plan, questionnaire, analysis

Technical Enrichment Activity #7 (User Study / Field Study). Deliverables required: study plan,
questionnaire, feedback from ≥30 participants, analysis, improvements incorporated.
**Get the guide's approval for this plan before sending the form.**

---

## 1. Study plan

**Goal.** Find out whether people who read financial reports can use FinSight, and whether showing
*how* an answer was produced (equation, unit, highlighted evidence, confidence) makes them trust and
check answers better than a bare answer. This tests the claims of Paper 2 with real users.

**Research questions**
- RQ1 Usability: is the demo easy to use? (System Usability Scale, SUS)
- RQ2 Explainability: does the evidence/equation view help users verify an answer and catch a wrong one?
- RQ3 Trust: do users prefer an explained answer, and do they want the system to say "not sure"?
- RQ4 Needs: what features matter for Indian users (lakh/crore, scanned PDFs, long reports)?

**Participants (target 35–40 so ≥30 complete).** Commerce/BBA/MBA/finance students, CA/CMA/CS
students, people working in finance/accounts/banking, and CSE students who know AI tools (cap this
group at ~⅓ so the results are not just from classmates). Record role in the form so groups can be compared.

**Procedure (~10–12 min per person)**
1. Consent + profile (Section A–B).
2. Participant opens the demo (Colab/Gradio link, or the team runs it for them in person/on a call).
3. Three fixed tasks on the same TAT-DQA test page (Section C). Use one span, one arithmetic,
   one where the system is **wrong** (pick one from the test run where FinSight's answer ≠ gold) —
   this tells us whether the evidence view helps people catch errors.
4. Screenshot comparison: bare answer vs. explained answer (Section D).
5. SUS + trust items + feature wishes + open feedback (Sections E–H).

**Materials to prepare**
- Demo link that stays up for the collection week (or run it live in sessions).
- Task page: the APAC total-assets page (Paper 3, Fig. 2) works well; take 3 questions from it.
- Two screenshots for Section D: **Screenshot A** = only "Answer: 21,234"; **Screenshot B** = answer +
  unit (thousand) + equation `958,744 − 937,510` + highlighted APAC row + confidence.
- Fill in the correct answers for tasks in a private answer key (not in the form).

**Ethics.** Anonymous; no names or emails required (make "collect emails" OFF in Google Forms);
participation is voluntary; data used only for the project report and papers.

**Timeline.** Pilot with 3 people → fix confusing questions → collect for 7–10 days → analyse →
make improvements → report.

---

## 2. Questionnaire (paste into Google Forms, one section per heading)

**Form-building rules** (so `scripts/analyse_user_study.py` can read the export):
keep the question wording as written · grids (Sections E and F) use columns labelled **1 2 3 4 5**,
not words · don't put ", " inside checkbox options · Settings → "Collect email addresses" OFF ·
mark every question Required except 7 and 44.

Form title: **FinSight: AI Question Answering on Financial Reports – User Feedback**

Form description:
> We are final-year CSE students at Amrita School of Computing, Coimbatore. FinSight answers questions
> about company financial reports (e.g., "What was the change in total assets?") and shows the
> calculation and the exact rows on the page that it used. This 10-minute survey helps us evaluate it.
> Responses are anonymous and used only for academic research.

### Section A: Consent
1. I am 18 or older and agree that my anonymous responses may be used for academic research. *(Required; checkbox: "I agree")*

### Section B: About you
2. Which best describes you? *(Multiple choice)*
   - UG student – Commerce / BBA / Economics
   - UG student – Engineering / Computer Science
   - PG student – MBA / M.Com / Finance
   - CA / CMA / CS student or article assistant
   - Working professional – finance, accounts, audit or banking
   - Working professional – other field
   - Other: ___
3. How often do you read financial statements or annual reports? *(Multiple choice)*
   - Never · A few times a year · Monthly · Weekly · Daily
4. How confident are you reading a balance sheet or income statement? *(Linear scale 1–5: Not at all → Very confident)*
5. How often do you use AI tools (ChatGPT, Gemini, Copilot, etc.)? *(Multiple choice)*
   - Never · Rarely · Weekly · Daily
6. Have you ever asked an AI tool a question about a financial document or numbers? *(Yes / No)*
7. *(If yes)* Was the answer correct? *(Multiple choice)*
   - Always correct · Mostly correct · Sometimes wrong · Often wrong · I didn't check

### Section C: Try FinSight (three tasks)
Section description: *Open the demo here: [LINK]. The report page is already loaded. Ask each question
below exactly as written, look at the answer and the highlighted page, then answer.*

**Task 1 – "What was the total assets from AMER in 2018?"**
8. What answer did FinSight give? *(Short answer)*
9. Is the answer correct? Check it against the page. *(Correct / Wrong / Not sure)*
10. Did the highlighted row help you check it? *(Linear scale 1–5: Not at all → A lot)*

**Task 2 – "What was the change in the total assets from APAC between 2018 and 2019?"**
11. What answer did FinSight give, including the unit? *(Short answer)*
12. Is the answer correct? *(Correct / Wrong / Not sure)*
13. Did seeing the equation (e.g., 958,744 − 937,510) help you check it? *(1–5: Not at all → A lot)*

**Task 3 – [question where FinSight is wrong; fill in]**
14. What answer did FinSight give? *(Short answer)*
15. Is the answer correct? *(Correct / Wrong / Not sure)*
16. What did you use to decide? *(Checkboxes)*
   - The highlighted rows on the page · The equation · The unit · The confidence score ·
     My own reading of the page · I guessed
17. How long did the three tasks take in total? *(Multiple choice: <2 min · 2–5 min · 5–10 min · >10 min)*

### Section D: Bare answer vs. explained answer
Section description: *Both screenshots answer the same question. [insert Screenshot A] [insert Screenshot B]*

18. Which answer do you trust more? *(Multiple choice: A – answer only · B – answer with equation and evidence · No difference)*
19. How much do you trust Answer A? *(1–5: Not at all → Completely)*
20. How much do you trust Answer B? *(1–5: Not at all → Completely)*
21. In B, which parts are useful? *(Checkboxes)*
   - Highlighted evidence rows · Equation · Unit / scale (thousand, million, %) · Confidence score · None

### Section E: Usability (System Usability Scale)
*Keep the wording exactly as below; it's a standard scale. Grid: rows = statements, columns 1–5
(Strongly disagree → Strongly agree).*

22. I think that I would like to use this system frequently.
23. I found the system unnecessarily complex.
24. I thought the system was easy to use.
25. I think that I would need the support of a technical person to be able to use this system.
26. I found the various functions in this system were well integrated.
27. I thought there was too much inconsistency in this system.
28. I would imagine that most people would learn to use this system very quickly.
29. I found the system very cumbersome to use.
30. I felt very confident using the system.
31. I needed to learn a lot of things before I could get going with this system.

### Section F: Trust and explanations
*Grid, 1–5 (Strongly disagree → Strongly agree)*

32. The highlighted evidence made it easy to check where the answer came from.
33. The equation helped me understand how the number was calculated.
34. Showing the unit (thousand / million / %) is important to me.
35. I would rather the system say "I'm not sure" than give a possibly wrong answer.
36. I would check the page myself before using an AI answer in real work (assignment, audit, report).
37. To show you are reading carefully, please select "Agree" for this row. *(attention check)*
38. I would trust FinSight more than a general chatbot (e.g., ChatGPT) for questions on financial reports.

### Section G: What you need
39. Which features would be most useful to you? *(Checkboxes, max 3)*
   - Indian number format (lakh/crore and 1,00,000 grouping)
   - Scanned / photographed reports (not just digital PDFs)
   - Long reports (full 100+ page annual report)
   - Questions in Hindi, Tamil or other Indian languages
   - Comparing two companies or two years
   - Export answers and evidence to Excel/PDF
   - Follow-up questions in a chat
   - Other: ___
40. What would you use FinSight for? *(Checkboxes)*
   - Studying / assignments · Exam (CA/CMA) preparation · Investing decisions · Audit or accounting work ·
     Research · Would not use it
41. Overall, how likely are you to recommend FinSight to a friend or colleague? *(Linear scale 0–10)*

### Section H: Open feedback
42. What was confusing or frustrating? *(Paragraph)*
43. What one change would make FinSight more useful for you? *(Paragraph)*
44. Any other comments? *(Paragraph, optional)*

---

## 3. Analysis plan (do this once responses are in)

Export responses to Google Sheets → CSV. Drop responses that fail Q37 (attention check) or are
duplicates; report how many were dropped.

| What | How | Answers |
|---|---|---|
| Participant profile | counts and % for Q2–Q7 | who took part |
| Task accuracy | % "Correct" judgements on Tasks 1–2; % who said **Wrong** on Task 3 | RQ2: can users verify/catch errors |
| Error-catching vs. evidence use | Task 3 detection rate split by whether Q16 includes "highlighted rows" or "equation" | RQ2 |
| SUS | per person: odd items (score − 1), even items (5 − score), sum × 2.5 → 0–100. Report mean ± SD. 68 = average; ≥ 80 = excellent | RQ1 |
| A vs. B trust | Q19 vs. Q20: median of each, Wilcoxon signed-rank test (paired); % choosing B in Q18 | RQ3 |
| Trust items Q32–Q38 | % agree (4–5) and median per item, stacked bar chart | RQ3 |
| Feature priorities | ranked counts for Q39 | RQ4 |
| NPS (Q41) | % promoters (9–10) − % detractors (0–6) | overall |
| Open answers | group into themes (e.g., "slow", "unit unclear", "want lakh/crore"), count each | improvements |
| By group | compare SUS and Q20 between finance and non-finance participants (Mann–Whitney U) | does background matter |

Everything in this table except the open-answer themes is computed by
`scripts/analyse_user_study.py` (already written): `python scripts/analyse_user_study.py responses.csv`

## 4. Improvements loop (needed for marks)

Write a table: *feedback theme → how many said it → change made → where in the code/demo*.
Likely candidates you can actually build:
- Show the scale next to the number in words ("21,234 **thousand** = 21.2 million").
- Indian format toggle (₹, lakh/crore, 1,00,000 grouping) in `demo.py` output.
- An "I'm not sure" message when votes disagree or operands aren't on the page (already have the signals in Paper 2).
- Clearer highlight colours / zoom to the evidence row.

If you re-test a change with even 5–10 people, add a before/after row — that's strong evidence.

## 5. Evidence to submit
Guide approval · this plan · form link + PDF print of the form · response sheet (anonymised CSV) ·
analysis notebook/script + charts · improvements table with screenshots before/after.

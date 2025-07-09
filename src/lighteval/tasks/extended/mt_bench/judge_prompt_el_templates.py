# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


def original_judge_prompt_mt_bench_el_without_ref(question, options, answer, gold):
    return [
        {
            "role": "user",
            "content": f"""# GOAL
Your job is to act as an impartial judge and to evaluate the quality of the response provided by an AI assistant to the user question.


You will be provided with the whole conversation in the Greek language, as well as the evaluation criteria. \
Your task is to evaluate the output of the AI system based on the evaluation criteria. \
Your evaluation should focus on the assistant's answer to the second user question.

<|The Start of Assistant A's Conversation with User|>\n\n\
### User:\n{question[0]}\n\n\
### Assistant A:\n{answer[0]}\n\n\
### User:\n{question[1]}\n\n\
### Assistant A:\n{answer[1]}\n\n\
<|The End of Assistant A's Conversation with User|>

# EVALUATION CRITERIA
All evaluation criteria should consider the fact that both the input and the response answer \
are in Greek and should be evaluated based on their score in Greek. \
Here are the evaluation criteria that you need to use for evaluating the task. \
<evaluation_criteria>
Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response.
</evaluation_criteria>

# INSTRUCTIONS FOR THE EVALUATION
Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, \
you must rate the response on a scale of 1 to 10. Then return your rating within <score> tags by strictly following this format: \"Rating: <score>rating</score>\", \
for example a score of 5 should be returned as: \"Rating: <score>5</score>\".\n\n"

Please accurately evaluate the task. Strictly adhere to the evaluation criteria.""",
        }
    ]


def original_judge_prompt_mt_bench_el_with_ref(question, options, answer, gold):
    return [
        {
            "role": "user",
            "content": f"""# GOAL
Your job is to act as an impartial judge and to evaluate the quality of the response provided by an AI assistant to the user question.

You will be provided with the inputs and output of the task in the Greek language, as well as the evaluation criteria. \
Your task is to evaluate the output of the AI system based on the evaluation criteria. \
Your evaluation should focus on the assistant's answer to the second user question.

<|The Start of Assistant A's Conversation with User|>\n\n\
### User:\n{question[0]}\n\n\
### Assistant A:\n{answer[0]}\n\n\
### User:\n{question[1]}\n\n\
### Assistant A:\n{answer[1]}\n\n\
<|The End of Assistant A's Conversation with User|>

# EVALUATION CRITERIA
All evaluation criteria should consider the fact that the input, response answer and \
reference answer are all in Greek and should be evaluated based on their score in Greek. \
Here are the evaluation criteria that you need to use for evaluating the task:
<evaluation_criteria>
How well the response answers the question, the reference answer is:
<|The Start of Reference Answer|>\n\n\
### User:\n{question[0]}\n\n\
### Reference answer:\n{gold[0]}\n\n\
### User:\n{question[1]}\n\n\
### Reference answer:\n{gold[1]}\n\n\
<|The End of Reference Answer|>
</evaluation_criteria>

# INSTRUCTIONS FOR THE EVALUATION
Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, \
you must rate the response on a scale of 1 to 10. Then return your rating within <score> tags by strictly following this format: \"Rating: <score>rating</score>\", \
for example a score of 5 should be returned as: \"Rating: <score>5</score>\".\n\n"

Please accurately evaluate the task. Strictly adhere to the evaluation criteria.""",
        }
    ]


def flow_judge_prompt_mt_bench_el_without_ref(question, options, answer, gold):
    return [
        {
            "role": "user",
            "content": f"""# GOAL
Your job is to evaluate a task carried out by an AI system powered by a large \
language model.

You will be provided with the inputs and output of the task in the Greek language,
as well as the evaluation criteria and scoring rubric. \
Your task is to evaluate the output of the AI system based on the evaluation \
criteria and scoring rubric provided.

# INPUT
Below are the inputs required for performing the task:
<inputs>
{question}
</inputs>

# OUTPUT
Below is the output of the task:
<output>
{answer}
</output>

# EVALUATION CRITERIA AND SCORING RUBRIC
Here are the evaluation criteria and the rubric that you need to use for evaluating the task. \
All evaluation criteria should consider the fact that both the input and the response answer \
are in Greek and should be evaluated based on their score in Greek:
<evaluation_criteria>
How well the response answers the question?
</evaluation_criteria>

<scoring_rubric>
- Score 1: The response completely fails to answer the question.
- Score 2: The response barely answers the question.
- Score 3: The response partially answers the question.
- Score 4: The response mostly answers the question.
- Score 5: The response completely answers the question.
</scoring_rubric>

# INSTRUCTIONS FOR THE EVALUATION
1. Understand the task and criteria: Familiarize yourself with the task to be evaluated. \
Review the evaluation criteria and scoring rubric to understand the different levels of \
performance and the descriptions for each score.
2. Review the inputs and output: Look at the inputs provided for the task. Examine the output \
generated from completing the task.
3. Compare output to score descriptions: Compare the output against the criteria and score \
descriptions in the scoring rubric. For each criterion,decide which description best matches the \
output.
4. After comparing the output to the score descriptions, pay attention to the small details that \
might impact the final score that you assign. Sometimes a small difference can dictate the final \
score.
5. Write verbal feedback justifying your evaluation that includes a detailed rationale, referring \
to specific aspects of the output and comparing them to the rubric.
6. Assign a final score based on the scoring rubric.

## FORMAT FOR THE EVALUATION
- Write the verbal feedback inside <feedback> tags without any additional surrounding text.
- Write the numeric score inside <score> tags, without any additional surrounding text and always \
after the feedback.

Please accurately evaluate the task. Strictly adhere to the evaluation criteria and rubric.""",
        }
    ]


def flow_judge_prompt_mt_bench_el_with_ref(question, options, answer, gold):
    return [
        {
            "role": "user",
            "content": f"""# GOAL
Your job is to evaluate a task carried out by an AI system powered by a large \
language model.

You will be provided with the inputs and output of the task in the Greek language, \
as well as the evaluation criteria and scoring rubric. \
Your task is to evaluate the output of the AI system based on the evaluation \
criteria and scoring rubric provided.

# INPUT
Below are the inputs required for performing the task:
<inputs>
{question}
</inputs>

# OUTPUT
Below is the output of the task:
<output>
{answer}
</output>

# EVALUATION CRITERIA AND SCORING RUBRIC
Here are the evaluation criteria and the rubric that you need to use for evaluating the task. \
All evaluation criteria should consider the fact that the input, response answer and \
reference answer are all in Greek and should be evaluated based on their score in Greek:
<evaluation_criteria>
How well the response answers the question, the reference answer is:
{gold}
</evaluation_criteria>

<scoring_rubric>
- Score 1: The response completely fails to answer the question.
- Score 2: The response barely answers the question.
- Score 3: The response partially answers the question.
- Score 4: The response mostly answers the question.
- Score 5: The response completely answers the question.
</scoring_rubric>

# INSTRUCTIONS FOR THE EVALUATION
1. Understand the task and criteria: Familiarize yourself with the task to be evaluated. \
Review the evaluation criteria and scoring rubric to understand the different levels of \
performance and the descriptions for each score.
2. Review the inputs and output: Look at the inputs provided for the task. Examine the output \
generated from completing the task.
3. Compare output to score descriptions: Compare the output against the criteria and score \
descriptions in the scoring rubric. For each criterion,decide which description best matches the \
output.
4. After comparing the output to the score descriptions, pay attention to the small details that \
might impact the final score that you assign. Sometimes a small difference can dictate the final \
score.
5. Write verbal feedback justifying your evaluation that includes a detailed rationale, referring \
to specific aspects of the output and comparing them to the rubric.
6. Assign a final score based on the scoring rubric.

## FORMAT FOR THE EVALUATION
- Write the verbal feedback inside <feedback> tags without any additional surrounding text.
- Write the numeric score inside <score> tags, without any additional surrounding text and always \
after the feedback.

Please accurately evaluate the task. Strictly adhere to the evaluation criteria and rubric.""",
        }
    ]


def flow_judge_prompt_mt_bench_el_without_ref_greek_judge(question, options, answer, gold):
    return [
        {
            "role": "user",
            "content": f"""# ΣΤΟΧΟΣ
Πρέπει να αξιολογήσεις την εργασία που επιτελεί ένα σύστημα AI το οποίο βασίζεται σε ένα μεγάλο γλωσσικό μοντέλο.
Θα σου δοθούν τα δεδομένα εισόδου και το αποτέλεσμα της εργασίας, καθώς και τα κριτήρια αξιολόγησης και η κλίμακα βαθμολόγησης. Η δική σου εργασία είναι να αξιολογήσεις το αποτέλεσμα του συστήματος AI με βάση τα κριτήρια αξιολόγησης και την κλίμακα βαθμολόγησης που θα παρασχεθεί.
# ΔΕΔΟΜΕΝΑ ΕΙΣΟΔΟΥ
Παρακάτω δίνονται τα δεδομένα εισόδου που απαιτούνται για την εκτέλεση της εργασίας:
<inputs>
{question}
</inputs>
# ΑΠΟΤΕΛΕΣΜΑ
Παρακάτω δίνεται το αποτέλεσμα:
<output>
{answer}
</output>
# ΚΡΙΤΗΡΙΑ ΑΞΙΟΛΟΓΗΣΗΣ ΚΑΙ ΚΛΙΜΑΚΑ ΒΑΘΜΟΛΟΓΗΣΗΣ
Παρακάτω δίνονται τα τα κριτήρια αξιολόγησης και η κλίμακα βαθμολόγησης που πρέπει να χρησιμοποιήσεις για την αξιολόγηση της εργασίας:
<evaluation_criteria>
Πόσο καλά απαντά η απάντηση του συστήματος στην ερώτηση;
</evaluation_criteria>
<scoring_rubric>
- Σκορ 1: Η απάντηση του συστήματος δεν απαντά στην ερώτηση.
- Σκορ 2: Η απάντηση του συστήματος απαντά οριακά στην ερώτηση.
- Σκορ 3: Η απάντηση του συστήματος απαντά εν μέρει στην ερώτηση.
- Σκορ 4: Η απάντηση του συστήματος απαντά σε μεγάλο βαθμό στην ερώτηση.
- Σκορ 5: Η απάντηση του συστήματος απαντά πλήρως στην ερώτηση.
</scoring_rubric>
# ΟΔΗΓΙΕΣ ΓΙΑ ΤΗΝ ΑΞΙΟΛΟΓΗΣΗ
1. Κατανόησε την εργασία και τα κριτήρια: Εξοικειώσου με την εργασία που πρόκειται να αξιολογηθεί. Εξέτασε τα κριτήρια αξιολόγησης και την κλίμακα βαθμολόγησης για να κατανοήσεις τα διαφορετικά επίπεδα απόδοσης και τις περιγραφές που συνοδεύουν κάθε σκορ.
2. Εξέτασε προσεκτικά τα δεδομένα εισόδου και το αποτέλεσμα: Εξέτασε τα δεδομένα εισόδου που παρέχονται για την εργασία. Εξέτασε το αποτέλεσμα που προέκυψε από την εκτέλεση της εργασίας.
3. Σύγκρινε το αποτέλεσμα με τις περιγραφές των βαθμολογιών: Σύγκρινε το αποτέλεσμα με τα κριτήρια και τις περιγραφές των βαθμολογιών στην κλίμακα βαθμολόγησης. Για κάθε κριτήριο, αποφάσισε ποια περιγραφή ταιριάζει καλύτερα με το αποτέλεσμα.
4. Μετά τη σύγκριση του αποτελέσματος με τις περιγραφές βαθμολογίας, δώσε προσοχή στις μικρές λεπτομέρειες που μπορεί να επηρεάσουν τον τελικό βαθμό που θα αποδώσεις στο αποτέλεσμα. Μερικές φορές μια μικρή λεπτομέρεια ή διαφορά μπορεί να καθορίσει τον τελικό βαθμό.
5. Δώσε ανατροφοδότηση που αιτιολογεί την αξιολόγησή σου: παρουσίασε το λεπτομερές σκεπτικό πίσω από την αξιολόγηση, ανάφερε συγκεκριμένες πτυχές ή τμήματα του αποτελέσματος και σύγκρινε τα τμήματα αυτά με την κλίμακα βαθμολόγησης.
6. Δώσε έναν τελικό βαθμό με βάση την κλίμακα βαθμολόγησης.
## ΜΟΡΦΟΠΟΙΗΣΗ ΤΗΣ ΑΞΙΟΛΟΓΗΣΗΣ
1. Τύπωσε την ανατροφοδότηση για το αποτέλεσμα μέσα σε tags <feedback> χωρίς επιπλέον κείμενο έξω από τα tags.
2. Τύπωσε τον τελικό βαθμό μέσα σε tags <score>, χωρίς επιπλέον κείμενο έξω από τα tags και πάντα μετά την ανατροφοδότηση.
Παρακαλώ να αξιολογήσεις με ακρίβεια την εργασία. Τήρησε αυστηρά τα κριτήρια αξιολόγησης και την κλίμακα βαθμολόγησης.""",
        }
    ]


def flow_judge_prompt_mt_bench_el_with_ref_greek_judge(question, options, answer, gold):
    return [
        {
            "role": "user",
            "content": f"""# ΣΤΟΧΟΣ
Πρέπει να αξιολογήσεις την εργασία που επιτελεί ένα σύστημα AI το οποίο βασίζεται σε ένα μεγάλο γλωσσικό μοντέλο.
Θα σου δοθούν τα δεδομένα εισόδου και το αποτέλεσμα της εργασίας, καθώς και τα κριτήρια αξιολόγησης και η κλίμακα βαθμολόγησης. Η δική σου εργασία είναι να αξιολογήσεις το αποτέλεσμα του συστήματος AI με βάση τα κριτήρια αξιολόγησης και την κλίμακα βαθμολόγησης που θα παρασχεθεί.
# ΔΕΔΟΜΕΝΑ ΕΙΣΟΔΟΥ
Παρακάτω δίνονται τα δεδομένα εισόδου που απαιτούνται για την εκτέλεση της εργασίας:
<inputs>
{question}
</inputs>
# ΑΠΟΤΕΛΕΣΜΑ
Παρακάτω δίνεται το αποτέλεσμα:
<output>
{answer}
</output>
# ΚΡΙΤΗΡΙΑ ΑΞΙΟΛΟΓΗΣΗΣ ΚΑΙ ΚΛΙΜΑΚΑ ΒΑΘΜΟΛΟΓΗΣΗΣ
Παρακάτω δίνονται τα τα κριτήρια αξιολόγησης και η κλίμακα βαθμολόγησης που πρέπει να χρησιμοποιήσεις για την αξιολόγηση της εργασίας:
<evaluation_criteria>
Πόσο καλά απαντά η απάντηση του συστήματος στην ερώτηση, δεδομένου ότι μια απάντηση αναφοράς από έναν ειδικό είναι η εξής;
{gold}
</evaluation_criteria>
<scoring_rubric>
- Σκορ 1: Η απάντηση του συστήματος δεν απαντά στην ερώτηση.
- Σκορ 2: Η απάντηση του συστήματος απαντά οριακά στην ερώτηση.
- Σκορ 3: Η απάντηση του συστήματος απαντά εν μέρει στην ερώτηση.
- Σκορ 4: Η απάντηση του συστήματος απαντά σε μεγάλο βαθμό στην ερώτηση.
- Σκορ 5: Η απάντηση του συστήματος απαντά πλήρως στην ερώτηση.
</scoring_rubric>
## ΜΟΡΦΟΠΟΙΗΣΗ ΤΗΣ ΑΞΙΟΛΟΓΗΣΗΣ
1. Τύπωσε την ανατροφοδότηση για το αποτέλεσμα μέσα σε tags <feedback> χωρίς επιπλέον κείμενο έξω από τα tags.
2. Τύπωσε τον τελικό βαθμό μέσα σε tags <score>, χωρίς επιπλέον κείμενο έξω από τα tags και πάντα μετά την ανατροφοδότηση.
Παρακαλώ να αξιολογήσεις με ακρίβεια την εργασία. Τήρησε αυστηρά τα κριτήρια αξιολόγησης και την κλίμακα βαθμολόγησης.""",
        }
    ]

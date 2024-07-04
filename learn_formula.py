from generate_formula import *
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Iterator, Tuple, List
from tqdm import tqdm


def generate_batched_question_answer_pairs(
    num_variables: int, num_clauses: int, num_batches: int, batch_size: int
) -> Iterator[Tuple[List[List[int]], List[bool]]]:
    """
    Generate an iterator of batched question-answer pairs for SAT formulas.

    Args:
        num_variables (int): The number of variables in the SAT formula.
        num_clauses (int): The number of clauses in the SAT formula.
        num_batches (int): The number of batches to generate.
        batch_size (int): The number of question-answer pairs in each batch.

    Yields:
        Iterator[Tuple[List[List[int]], List[bool]]]: An iterator of batched question-answer pairs, where each batch contains
                                                      a list of questions (variable assignments) and a list of corresponding answers.
    """
    formula = generate_random_sat_formula(num_variables, num_clauses)

    for _ in range(num_batches):
        batch_questions = []
        batch_answers = []
        for _ in range(batch_size):
            assignment = [random.choice([True, False]) for _ in range(num_variables)]
            answer = evaluate_formula(formula, assignment)
            batch_questions.append(assignment)
            batch_answers.append(answer)
        yield batch_questions, batch_answers


def evaluate_model_probability(
    questions: List[List[bool]],
    answers: List[bool],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
) -> TensorType["batch_size"]:
    """
    Evaluate the probability that the language model assigns to the correct answers given the questions in a batched manner.

    Args:
        questions (List[List[bool]]): A list of questions, where each question is a list of variable assignments.
        answers (List[bool]): A list of correct answers (True or False) corresponding to each question.
        model (AutoModelForCausalLM): The language model.
        tokenizer (AutoTokenizer): The tokenizer for the language model.

    Returns:
        TensorType["batch_size"]: The probabilities assigned to the correct answers for each question in the batch.
    """
    prompts = [
        f"""Given a Boolean satisfiability (SAT) problem with the following variable assignments:
{question}
Is the SAT formula satisfied? Answer with 0 for False or 1 for True.
Answer: """
        for question in questions
    ]

    # Tokenize all prompts
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]

    # Get the logits for the '0' and '1' tokens
    zero_token = tokenizer.encode("0")[-1]
    one_token = tokenizer.encode("1")[-1]
    zero_one_logits = logits[:, [zero_token, one_token]]

    # Apply softmax to get probabilities for '0' and '1'
    zero_one_probs = torch.softmax(zero_one_logits, dim=-1)

    # Get the probabilities of the correct answers
    # correct_answer_indices = torch.tensor(answers, dtype=torch.long)
    probabilities_of_truth = zero_one_probs[torch.arange(len(answers)), 1]
    return probabilities_of_truth


def test_evaluate_model_probability():
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    # Test data
    questions = [[True, False, True], [False, True, False]]
    answers = [True, False]

    # Run the function
    probabilities = evaluate_model_probability(questions, answers, model, tokenizer)

    # Check the output
    assert isinstance(probabilities, torch.Tensor)
    assert probabilities.shape == torch.Size([2])  # Batch size of 2
    assert (probabilities >= 0).all() and (
        probabilities <= 1
    ).all()  # Probabilities should be between 0 and 1

    print("Test passed successfully!")


def train_model(model, tokenizer, optimizer, dataset_iterator, num_batches):
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    progress_bar = tqdm(
        dataset_iterator, total=num_batches, desc="Training", unit="batch"
    )

    for questions, answers in progress_bar:
        # Initialize loss variable
        loss = None
        # Prepare input prompts
        prompts = [
            f"Given a hidden Boolean satisfiability (SAT) problem with the following variable assignments: {question}\nIs the SAT formula satisfied? Answer with 0 for False or 1 for True.\nAnswer: "
            for question in questions
        ]

        # Tokenize inputs
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        # Prepare target outputs
        targets = torch.tensor(
            [int(answer) for answer in answers], dtype=torch.long
        ).to(device)

        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]  # Get logits for the last token

        # Extract logits for '0' and '1' tokens
        zero_token = tokenizer.encode("0")[-1]
        one_token = tokenizer.encode("1")[-1]
        zero_one_logits = logits[:, [zero_token, one_token]]

        # Apply softmax to get probabilities for '0' and '1'
        zero_one_probs = torch.softmax(zero_one_logits, dim=-1)

        # Calculate the probability assigned to the correct answer
        correct_probs = zero_one_probs[torch.arange(len(targets)), targets]

        # Calculate the probability assigned to the wrong answer (1 - correct probability)
        wrong_probs = 1 - correct_probs

        # Use wrong probabilities as loss
        loss = wrong_probs.mean()
        ## Calculate loss based on the normalized probabilities
        # assert loss == torch.nn.functional.cross_entropy(zero_one_probs, targets)
        # loss = torch.nn.functional.cross_entropy(zero_one_probs, targets)
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # progress_bar.set_postfix(loss=loss.item())
        bar_length = 20
        loss_value = loss.item()
        num_bars = int(loss_value * bar_length)
        loss_bar = "[" + "#" * num_bars + "-" * (bar_length - num_bars) + "]"
        # Update the progress bar with the current loss and ASCII bar plot
        progress_bar.set_postfix(loss=loss_value, loss_bar=loss_bar)


def test_train_model():
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Mock model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to(device)
    # Set up optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # Training parameters
    num_variables = 4
    num_clauses = 5
    num_batches = 100
    batch_size = 64

    # Create dataset iterator
    dataset_iterator = generate_batched_question_answer_pairs(
        num_variables, num_clauses, num_batches, batch_size
    )

    # Run the training function
    train_model(model, tokenizer, optimizer, dataset_iterator, num_batches)

    print("Test passed successfully!")


# Run the test
test_train_model()

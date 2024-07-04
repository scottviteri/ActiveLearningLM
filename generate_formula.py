import random
from typing import List
import pytest
from torchtyping import TensorType
import torch


def generate_random_sat_formula(
    num_variables: int, num_clauses: int
) -> List[List[int]]:
    """
    Generate a random SAT formula with the given number of variables and clauses.

    Args:
        num_variables (int): The number of variables in the SAT formula.
        num_clauses (int): The number of clauses in the SAT formula.

    Returns:
        List[List[int]]: A list of clauses, where each clause is a list of integers representing literals.
    """
    formula = []
    for _ in range(num_clauses):
        clause = []
        clause_size = random.randint(
            1, num_variables
        )  # Random clause size between 1 and num_variables
        for _ in range(clause_size):
            literal = random.randint(1, num_variables)  # Random variable
            if random.choice([True, False]):
                literal = -literal  # Randomly negate the literal
            clause.append(literal)
        formula.append(clause)
    return formula


def test_generate_random_sat_formula():
    num_variables = 5
    num_clauses = 3
    formula = generate_random_sat_formula(num_variables, num_clauses)

    assert (
        len(formula) == num_clauses
    ), f"Expected {num_clauses} clauses, but got {len(formula)}"

    for clause in formula:
        assert (
            1 <= len(clause) <= num_variables
        ), f"Clause size {len(clause)} is out of bounds"
        for literal in clause:
            assert (
                -num_variables <= literal <= num_variables and literal != 0
            ), f"Literal {literal} is out of bounds"


def evaluate_formula(formula: List[List[int]], assignment: List[bool]) -> bool:
    """
    Evaluate the truth of a SAT formula given a variable assignment.

    Args:
        formula (List[List[int]]): The SAT formula to evaluate.
        assignment (List[bool]): The variable assignment.

    Returns:
        bool: True if the formula is satisfied, False otherwise.
    """
    for clause in formula:
        clause_satisfied = False
        for literal in clause:
            var_index = abs(literal) - 1
            var_value = assignment[var_index]
            if literal < 0:
                var_value = not var_value
            if var_value:
                clause_satisfied = True
                break
        if not clause_satisfied:
            return False
    return True


def test_evaluate_formula():
    # Test case 1: Simple satisfiable formula
    formula1 = [[1, 2], [-1, 3]]
    assignment1 = [True, False, True]
    assert evaluate_formula(formula1, assignment1) == True

    # Test case 2: Simple unsatisfiable formula
    formula2 = [[1], [-1]]
    assignment2 = [True]
    assert evaluate_formula(formula2, assignment2) == False
    assignment3 = [False]
    assert evaluate_formula(formula2, assignment3) == False

    # Test case 3: More complex formula
    formula3 = [[1, 2, 3], [-1, -2], [2, -3]]
    assignment4 = [True, True, False]
    assert evaluate_formula(formula3, assignment4) == False
    assignment5 = [False, False, True]
    assert evaluate_formula(formula3, assignment5) == False

    # Test case 4: Empty formula (always true)
    formula4: List[List[int]] = []
    assignment6: List[bool] = []
    assert evaluate_formula(formula4, assignment6) == True

    # Test case 5: Formula with empty clause (always false)
    formula5 = [[1, 2], []]
    assignment7 = [True, True]
    assert evaluate_formula(formula5, assignment7) == False


def test_evaluate_formula_with_tensors():
    # Test case with tensor inputs
    formula_tensor: TensorType["num_clauses", "max_literals"] = torch.tensor(
        [
            [1],
        ]
    )
    assignment_tensor: TensorType["num_variables"] = torch.tensor([True])

    result = evaluate_formula(formula_tensor.tolist(), assignment_tensor.tolist())
    assert result == True


if __name__ == "__main__":
    import pytest

    # Run all test functions in this module
    pytest.main([__file__])

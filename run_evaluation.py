from evaluation.regression_paper_evaluation import paper_results_evaluation
from project_tests import test_paper_experiments


def main():
    # Process the logs
    test_paper_experiments.evaluate_results()

    # Prepare
    paper_results_evaluation()


if __name__ == "__main__":
    main()
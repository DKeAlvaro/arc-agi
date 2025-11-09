from utils import plot_pairs
from matplotlib.backends.backend_pdf import PdfPages
import os
import random

def make_predictions(test_challenges, model, solutions=None, verbose=False):
    """Runs the given model against all test challenges and prints accuracy."""
    all_challenge_ids = list(test_challenges.keys())
    print(f"Test challenges: {len(all_challenge_ids)}")
    count = 0

    pdf_pages = None
    plotting_challenge_ids = []
    if verbose:
        if not os.path.exists("predictions"):
            os.makedirs("predictions")
        pdf_pages = PdfPages(os.path.join("predictions", "all_predictions.pdf"))
        num_to_plot = min(5, len(all_challenge_ids))
        plotting_challenge_ids = random.sample(all_challenge_ids, num_to_plot)

    predictions = {}

    # Evaluation metrics
    total_correct = 0
    total_attempts = 0
    total_pixels_correct = 0
    total_pixels = 0

    for i, challenge_id in enumerate(all_challenge_ids):
        challenge_data = test_challenges[challenge_id]
        train_examples = challenge_data['train']
        test_examples = challenge_data['test']

        # Retrain model
        model.train(train_examples)

        # Generate model solutions for training data (for plotting)
        model_train_solutions = []
        for train_example in train_examples:
            model_solution = model.predict(train_example['input'])
            model_train_solutions.append(model_solution)

        # Plot training pairs with model solutions if verbose
        if verbose and challenge_id in plotting_challenge_ids:
            print(f"\nChallenge {challenge_id} - Training examples:")
            plot_pairs(train_examples, model_train_solutions, title=f"Challenge {challenge_id} - Training", pdf_pages=pdf_pages)

        challenge_predictions = []
        challenge_correct = 0
        challenge_attempts = 0
        challenge_pixels_correct = 0
        challenge_pixels = 0

        # Generate model solutions for test data (for plotting)
        model_test_solutions_attempt1 = []
        model_test_solutions_attempt2 = []

        for test_example in test_examples:
            input_data = test_example['input']

            pred1 = model.predict(input_data)
            pred2 = model.predict(input_data)

            attempts = {
                'attempt_1': pred1,
                'attempt_2': pred2
            }

            challenge_predictions.append(attempts)
            model_test_solutions_attempt1.append(pred1)
            model_test_solutions_attempt2.append(pred2)

        # Get true solutions for this challenge if available
        true_test_solutions = None
        if solutions and challenge_id in solutions:
            true_test_solutions = solutions[challenge_id]

        # Plot test pairs with model solutions if verbose
        if verbose and challenge_id in plotting_challenge_ids:
            print(f"Challenge {challenge_id} - Test examples (Attempt 1):")
            plot_pairs(test_examples, model_test_solutions_attempt1, true_test_solutions,
                      f"Challenge {challenge_id} - Test (Attempt 1)", pdf_pages=pdf_pages)

            print(f"Challenge {challenge_id} - Test examples (Attempt 2):")
            plot_pairs(test_examples, model_test_solutions_attempt2, true_test_solutions,
                      f"Challenge {challenge_id} - Test (Attempt 2)", pdf_pages=pdf_pages)

        predictions[challenge_id] = challenge_predictions

        # Evaluate if solutions are provided
        if solutions and challenge_id in solutions:
            for i, (predicted, solution) in enumerate(zip(challenge_predictions, solutions[challenge_id])):
                for attempt_name in ['attempt_1', 'attempt_2']:
                    total_attempts += 1
                    challenge_attempts += 1
                    pred_output = predicted[attempt_name]

                    if pred_output == solution:
                        total_correct += 1
                        challenge_correct += 1

                    try:
                      pred_flat = [pixel for row in pred_output if hasattr(row, '__iter__') for pixel in row]
                      sol_flat = [pixel for row in solution if hasattr(row, '__iter__') for pixel in row]
                    except:
                      continue

                    pixels_correct = sum(1 for p, s in zip(pred_flat, sol_flat) if p == s)
                    total_pixels_this = len(sol_flat)

                    total_pixels_correct += pixels_correct
                    total_pixels += total_pixels_this
                    challenge_pixels_correct += pixels_correct
                    challenge_pixels += total_pixels_this

        # Print challenge-level results if solutions provided
        if solutions and challenge_id in solutions:
            count += 1
            challenge_exact_acc = challenge_correct / challenge_attempts if challenge_attempts > 0 else 0
            challenge_pixel_acc = challenge_pixels_correct / challenge_pixels if challenge_pixels > 0 else 0
            print(f"{count}/{len(all_challenge_ids)} - {challenge_id}: Exact {challenge_exact_acc:.2%}, Pixel {challenge_pixel_acc:.2%}")

    # Print overall results if solutions provided
    if solutions:
        exact_accuracy = total_correct / total_attempts if total_attempts > 0 else 0
        pixel_accuracy = total_pixels_correct / total_pixels if total_pixels > 0 else 0

        print(f"\nOverall Accuracy: {exact_accuracy:.2%} ({total_correct}/{total_attempts})")
        print(f"Overall Pixel Accuracy: {pixel_accuracy:.2%}")

    if pdf_pages:
        pdf_pages.close()

    return predictions
from rouge_score import rouge_scorer

def rouge_scores(filename):
    # Read the text file
    with open(filename, "r") as f:
        content = f.read()

    # Split the content into test samples
    samples = content.split("\n\n")
    scorer = rouge_scorer.RougeScorer(['rouge2', 'rougeL'], use_stemmer=True)

    for sample in samples:
        lines = sample.strip().split("\n")
        generated_summary = lines[5].replace('<pad>', '').replace('<sos>', '').replace('<eof>', '')
        original_summary = lines[3].replace('<pad>', '').replace('<sos>', '').replace('<eof>', '')

        scores = scorer.score(generated_summary, original_summary)
        lines[1] = lines[1].replace("<eos>", "").replace("<pad>", "").replace("<sos>", "")
        with open(f'scores_{filename}', "a") as f:
            f.write(f"Rouge-2: {scores['rouge2']} \n")
            f.write("\n\n")

rouge_scores('results_topp.txt')
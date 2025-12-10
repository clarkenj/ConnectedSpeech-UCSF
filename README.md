# ConnectedSpeech-UCSF
Data processing and analysis of connected speech features in PPA

## Installation

1. Clone the repository

```shell
git clone -b main https://github.com/lingualab/ConnectedSpeech-UCSF
cd ConnectedSpeech-UCSF
```

2. Create a virtual environment

```shell
python3 -m venv --prompt ConnectedSpeech-UCSF venv
source venv/bin/activate
```

3. Install ConnectedSpeech-UCSF:

```shell
pip install -e .
```

## Tasks description
Participants completed the Picnic Scene from the Western Aphasia Battery. Five transcripts are available:
- `salt.slt`: Original SALT transcript.
- `salt.txt`: SALT transcript with all SALT codes removed.
- `manual.txt`: SALT transcript with all SALT codes removed, and new lines removed.
- `whisper.txt`: Original Whiser transcript.
- `whisperQC.txt`: Whisper transcript manually checked, and disfluencies added.

## Notes
- Updated data was sent from UCSF. We now use the data in `/data/brambati/dataset/ConnectedSpeech-UCSF/sourcedata/NEW`.

## Scripts
- `csucsf_process_merge`: merge select speech features output from [speechmetryflow](https://github.com/lingualab/speechmetryflow) with phenotype information for further analysis.
- `csucsf_describe_participants`: calculate demographic statistics based on data from included participants
- `csucsf_analysis_wer`: performs text analysis and compute Word Error Rates metric
- `csucsf_analysis_icc`: compute intraclass correlation analysis
- `csucsf_classification`: run binary classification for specified pairs of diagnoses, using linguistic features selected in the `csucsf_process_merge` script. Optionally include feature selection.
- `csucsf_classification_stats`: plot bar graphs comparing the classification performance of different transcription methods.

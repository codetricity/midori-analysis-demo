# Pandas AI Test

using camera360 user data

## python setup

* python must be 3.11.9.  will not work
with 3.12.*
* suggest using pyenv to setup python version

```text
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## install

* use requirements.txt
* the numpy, pandas and pandasai requirements are complex

## setup of security

* create `./streamlit/secrets.toml`

```text
password = "put_secret_password here"

[data]
gdrive_file_id = "put google sheets file id here"

[openai]
api_key = "sk-proj..."
```

## run

`streamlit run app.py`

# vae-reconstruction-comparison

## Setup Instructions

### Files

There is a file ***stud_data.csv*** which needs to be downloaded and placed into the folder **cvae_experiment**:

[Download stud_data.csv here](https://send.firefox.com/download/3e473ec785cb03a1/#iZ0ywWh_sCUGdowfAExr3A)

### Virtual Environment

1. Install virtualenv if you haven't already got it
```bash
pip3 install virtualenv
```
2. Navigate to the right directory

```bash
cd cvae_experiment
```
3. Create your virtual environment, name it venv
```bash
python3 -m virtualenv venv
```
4. Activate the virtualenv
```bash
source venv/bin/activate
# If this does not work, ensure you are in cvae_experiment directory
which pip3
which python3
# these should show that pip3 and python3 exist in your venv folder which means the virtualenv is working, otherwise something went wrong

```

5. Install required python3 packages
```bash
pip3 install -r requirements.txt
``` 

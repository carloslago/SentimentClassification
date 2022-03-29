# SentimentClassification

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Contributors](#contributors)
* [Usage](#usage)
* [Contributing](#contributing)
* [Contact](#contact)



<!-- ABOUT THE PROJECT -->
## About The Project

This is a university project developed for the "Data Science" course at KTH, for this project we aimed to perform sentiment classification of social media comments in swedish using semi-supervised learning. The final report can be seen [here](#Research_report.pdf).

### Built With

* [Python](https://www.python.org/)
* [Pytorch](https://pytorch.org/)
* [HuggingFace](https://huggingface.co/)

## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore -->
<table align="center">
  <tr>
    <td align="center"><a href="https://github.com/carloslago">
        <img src="https://avatars2.githubusercontent.com/u/15263623?s=400&v=4" 
        width="150px;" alt="Carlos Lago"/><br/><sub><b>Carlos Lago</b></sub></a><br/></td>
    <td align="center"><a href="https://github.com/xc-liu">
        <img src="https://avatars.githubusercontent.com/u/47290005?v=4" 
        width="150px;" alt="Xuecong Liu"/><br /><sub><b>Xuecong Liu</b></sub></a><br/></td>
   <td align="center"><a href="https://github.com/xc-liu">
        <img src="https://avatars.githubusercontent.com/u/43607124?v=4" 
        width="150px;" alt="Zhenlin Zhou"/><br /><sub><b>Zhenlin Zhou</b></sub></a><br/></td>
   <td align="center"><a href="https://github.com/eliott-remmer">
        <img src="https://avatars.githubusercontent.com/u/73662183?v=4" 
        width="150px;" alt="Xuecong Liu"/><br /><sub><b>Eliott Remmer</b></sub></a><br/></td>

  </tr>
</table>

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites
* Python
```sh
sudo apt install python3 python3-dev
```
### Installation
 
1. Clone the repo
```sh
git clone https://github.com/carloslago/SentimentClassification.git
```
2. Install Python packages
```sh
sudo pip install -r requirements.txt
```

## Usage

The project was oriented to test both the fine-tuning of a BERT model with social media comments and the training of a GAN-BERT model, making use of unsupervised data.

Data should have the following format, for unlabelled data the sentiment label doesn't need to be present.
```
{"sentiment_label": "-1/1", "message": "This is a comment in swedish"}
```

Fine-tuning swedish BERT:
```
python bert_fine_tuning.py
```

Training GAN-BERT with unsupervised data:
```
python gan_bert.py
```
<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

<!-- CONTACT -->
## Contact

Feel free to send us any bug reports, ask me any questions or request any features via email, just keep in mind we did this as a university project.
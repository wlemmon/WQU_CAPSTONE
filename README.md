# WQU CAPSTONE Project Winter 2023
This repo houses the code for my project related to piotroski fscore and the effects of market conditions of F-score portfolio performance.

# Installation
I build a conda virtual environment following these commands:
-conda install python=3.10
-pip install matplotlib
-pip install financetoolkit -U
-pip install financedatabase -U
-pip install jupyter
-pip install notebook
-pip install seaborn
-pip install dataframe_image

A dump of all installed packages from my environment can be found in requirements.txt. Running pip install –r requirements.txt should allow you to run the notebooks.
Run the notebooks by starting jupyter notebook from the main directory.

# Project Track
Quantitative Fundamentals - Applications of Piotroski F-Score. I will be doing research, less
application.

# Problem statement
What is the effect of market bubbles on Piotroski F-score? Piotroski F-Score (Piotroski) was originally
used to demonstrate excess returns for US stocks with high book-to-market ratios. In 2017 (Turtle)
the F-score was shown useful as a generic indicator of firm strength beyond value stocks. In 2020
(Walkshäusl), the F-score’s generic predictive power was further validated in out-of-sample markets
(Europe, Australasia, and Far East). However, other factor models have been studied under bubble
conditions. Fama & French 3 Factor models have been shown (Wang) to decline in explanatory power
during periods of bubbles. Also, markets are ever changing and Covid-19 presents additional
opportunities to study Piotroski FScore under bubble conditions. To the best of my knowledge, this
will be the first work to study the effects of market bubbles on Piotroski F-score performance as an
indicator of firm strength. It will also include data post Covid-19 which includes a unique market
bubble due to global government pandemic response.

# Goals and Objectives
Write your goals(s) and objectives here.
* Compare Fscore predictive power during bubbles with Prior work on Fama and French factor
models. Describe similarities and differences discovered.
** Compare results with study on bubbles pertaining to Fama and French factor model
(Wang). I think I will use similar statistical measures for bubble formation and
statistical significance as this will make results more consistent with this prior work.

* Measure bubble vs non-bubble predictive power.
* Measure predictive power for value stocks vs growth stocks.
* Remark on post Covid-19 bubble’s effects on Piotroski F-score explanatory power.
* Not limiting myself to China market only, include F-score predictive power in bubbles vs
non-bubbles in US market and possibly other emerging markets, time granting.

# Code Design
* Financial data will be collected from Yahoo Finance. For comparison with prior work, I may
obtain data from the same sources as those works for reproducibility.
* Analysis will be performed in python.
* Since I'm a solo worker on this Capstone, no collaboration is required.

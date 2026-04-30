### [ Practice Module ] Project Submission Template: Github Repository & Zip File

**[ Naming Convention ]** CourseCode-StartDate-BatchCode-TeamName-ProjectName.zip

* **[ MTech Thru-Train Group Project Naming Example ]** IRS-PM-2020-01-18-IS02PT-GRP-AwsomeSG-HDB_BTO_Recommender.zip

* **[ MTech Stackable Group Project Naming Example ]** IRS-PM-2020-01-18-STK02-GRP-AwsomeSG-HDB_BTO_Recommender.zip

[Online editor for this README.md markdown file](https://pandao.github.io/editor.md/en.html "pandao")

---

### <<<<<<<<<<<<<<<<<<<< Start of Template >>>>>>>>>>>>>>>>>>>>

---

## SECTION 1 : PROJECT TITLE
## <p align="justify"> Development of an Intelligent Platform for Automated Conjunctival Vessel Segmentation, Efron Severity Classification & LLM-Assisted Clinical Interpretation
</p>

## Section 2 : PROJECT SUMMARY

<p align="justify">
This project presents an end-to-end automated platform for conjunctival hyperaemia Efron severity grading, where conjunctival vessels are extracted automatically through a Semi-Supervised Learning segmentation model, severity is classified using Ordinal Logistic Regression, and clinical interpretation is generated automatically via a Large Language Model — all integrated into a single clinician-friendly Streamlit interface accessible to any clinician without requiring technical expertise.
</p>

<p align="justify">
Conjunctival hyperaemia is the redness of the white part of the eye which is a key clinical indicator of dry eye disease severity. Current grading practice relies on manual visual assessment by clinicians using the Efron grading scale (Grade 0 to Grade 4) , which is highly subjective and prone to inter-observer variability. This project addresses this gap by developing an intelligent platform that automates the entire grading pipeline and extraction of the vascular structure, reducing inter-observer variability, supporting faster diagnostic decisions and improving consistency in conjunctival hyperaemia grading in real-world ophthalmology settings.
</p>

<table>
  <tr>
    <td align="center"><strong>Images</strong></td>
    <td align="center"><strong>Reference Images for Efron severity grading</strong></td>
  </tr>
  <tr>
    <td><img src="System_workflow/images.png" width="450"/></td>
    <td><img src="System_workflow/reference_image.png" width="450"/></td>
  </tr>
</table>


## Section 3 : SYSTEM PIPELINE

### 3.1 Semi-Supervised Learning (SSL) Segmentation

<p align="justify">
The platform integrates three core components. First, a Semi-Supervised Learning (SSL) segmentation model based on a cross-teaching framework between UNet and Swin-UNet extracts conjunctival vascular structure from slit-lamp images, achieving an IoU of 0.522 with limited annotated data. The process begins with the original conjunctival slit-lamp image which undergoes Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance image contrast and improve vessel visibility. A conjunctival mask is then applied to isolate the region of interest. The masked image is then divided into 256×256 pixel patches using a sliding window approach with 50% overlap, generating both image patches and their corresponding binary vessel label patches. These patches are fed into the SSL segmentation model, which produces predicted binary vessel masks for each patch. uring inference, the best saved model is loaded and applied to the unseen image patches to produce predicted binary vessel masks for each patch. The predicted patches then undergo post-processing before being stitched back together to reconstruct the full binary vessel mask at the original image resolution of 1280×980 pixels. The reconstructed vessel mask is subsequently used to calculate the vessel density value, which is then used to generate a correlation plot of vessel density against the mean Efron severity grade for evaluation. A key advantage of the SSL approach is that the segmentation pipeline is fully automated — once trained, the model is capable of extracting conjunctival vessel structures directly from raw slit-lamp images without requiring any manual annotation or expert labelling, making it highly scalable and practical for real-world clinical deployment where annotated data is scarce and expensive to obtain. 
</p>

<table>
  <tr>
    <td align="center"><strong>SSL Model</strong></td>
    <td align="center"><strong>Process training SSL Model</strong></td>
  </tr>
  <tr>
    <td><img src="System_workflow/SSLModel.png" width="450"/></td>
    <td><img src="System_workflow/process_training_SSLModel.png" width="450"/></td>
  </tr>
</table>

### 3.2 Classification Model

<p align="justify">
Second, an Ordinal Logistic Regression classification model maps the extracted vessel density to an Efron severity grade. Ordinal Logistic Regression was selected as the classification model as it is specifically designed for ordered categorical outcomes, explicitly respecting the 
natural ordering of the Efron severity scale from Grade 0 to Grade 4, unlike standard multiclass classification models such as Support Vector Machine, Random Forest and K-Nearest Neighbours which treat each grade as an independent unordered category. The model was trained on a dataset 
of 633 images with balanced class weighting applied to account for the unequal distribution of samples across severity grades and evaluated on a separate held-out test set of 32 unseen images.

The model achieved an overall accuracy of 78.1% and a Pearson correlation of 0.934 between the predicted vessel density and the model-predicted Efron severity grades on the 32 unseen test images, surpassing the clinician ground truth benchmark correlation of 0.854. This suggests 
that the model has learned a highly consistent and systematic mapping between vessel density and Efron severity grade — one that is more internally consistent than human grading which is naturally subject to subjective inter-observer variability. The higher correlation observed 
between predicted vessel density and model-predicted grades compared to clinician grades further validates the use of vessel density as a reliable and objective biomarker for Efron severity classification. 
</p>

<table>
  <tr>
    <td align="center"><strong>Process training Ordinal Logistic Regression Classification Model</strong></td>
  </tr>
  <tr>
    <td align="center">
      <img src="System_workflow/OLRModel.png" width="450"/></td>
  </tr>
</table>

Third, Claude Sonnet v3.5 via AWS Bedrock is integrated as a clinical explainer, where the predicted grade and vessel density are passed into the Large Language Model (LLM) to generate a natural language clinical interpretation, management recommendation, lifestyle advice, follow-up actions and red flags for the clinician.
</p>


The system was initially designed with LLM-based grading in mind, where the LLM would predict the Efron Severity grade by visually comparing the input conjunctival image against reference images across three prompting strategies — Input-Output (IO), Chain-of-Thought (CoT), and Tree-of-Thought (ToT). However, upon evaluation, the LLM grading results were outperformed by the conventional Ordinal Logistic Regression classification model, which achieved a Pearson correlation of 0.934 compared to the best LLM correlation of 0.631 (Claude Sonnet v3.5, CoT). Based on these results, an evidence-based decision was made to repurpose the LLM as a clinical explainer rather than a primary grader which is a more appropriate and clinically valuable use of its capability. Hence the predicted grade from classification model is passed into the LLM to generate a natural language clinical interpretation, management recommendation and follow-up actions for the clinician. 
</p>

<p align="justify">
Due to the time constraints, fine-tuning of the LLM was not performed and by using the different prompt strategies such as IO, CoT and ToT alongside with the reference image was insufficient as the model like ChatGPT v4.0 and Claude Sonnet v3.5 relied entirely on prompt engineering and reference images without any task-specific adaptation. The method of prompting strategies significantly influenced the grading performance resulting in lower performance. However, LLM used to predict the Efron severity grades remains as the promising direction in future work.
</p>

<table>
  <tr>
    <td><img src="System_workflow/System_Pipeline_before.png" width="450"/></td>
    <td><img src="System_workflow/System_Pipeline_after.png" width="450"/></td>
  </tr>
</table>


---

## SECTION 2 : EXECUTIVE SUMMARY / PAPER ABSTRACT
Singapore ranks amongst countries with the highest population density in the world. In a bid to have firm control over long term urban planning, the Singapore government came up with the “Built to Order” (abbreviated BTO) initiative back in 2001. These are new Housing Development Board (HDB) flats tightly controlled by their eligibility and quantity released every year. In more recent years, the modern BTO scheme in Singapore requires a waiting period of 3-4 years, and is generally targeted at young Singaporean couples looking to purchase their first property and set up a family. Nationality and income ceilings are some of the broad filters that determine one’s eligibility for the highly sought after projects. 


Our team, comprising of 6 young Singaporeans, all hope to be property owners one day. Many of our peers opt for BTO flats due to their affordability, existence of financial aid from the government, as well as their resale value. However, there often exists a knowledge gap for these young couples during the decision making process and they end up making potentially regretful decisions. We would like to bridge this knowledge gap, and have hence chosen to base our project on creating a recommender system for BTO flats, utilizing the data from recent launches in Tampines, Eunos, Sengkang and Punggol. 


Using the techniques imparted to us in lectures, our group first set out to build a sizeable knowledge base via conducting an interview and administering a survey. While building the system, we utilized tools such as Java to scrape real time data from HDB website and transform it into a database, CLIPS to synthesize the rule based reasoning process, and Python to integrate it into an easy to use UI for the everyday user. To add icing on the cake, we even hosted the system on a website so that the everyday user can access it through the click of a link.


Our team had an amazing time working on this project, and hope to share our insights with everyone. Despite a focus on BTO flats, we would recommend it for everybody interested in understanding property market trends for residence or investment purposes. There truly are a wide array of factors behind the decision to invest in a property, and we only wish there was more time to work on the scope and scale of the project. 

---

## SECTION 3 : CREDITS / PROJECT CONTRIBUTION

| Official Full Name  | Student ID (MTech Applicable)  | Work Items (Who Did What) | Email (Optional) |
| :------------ |:---------------:| :-----| :-----|
| Desmond Chua | A1234567A | xxxxxxxxxx yyyyyyyyyy zzzzzzzzzz| A1234567A@nus.edu.sg |
| Chang Ye Han | A1234567B | xxxxxxxxxx yyyyyyyyyy zzzzzzzzzz| A1234567B@gmail.com |
| Chee Jia Wei | A1234567C | xxxxxxxxxx yyyyyyyyyy zzzzzzzzzz| A1234567C@outlook.com |
| Ganesh Kumar | A1234567D | xxxxxxxxxx yyyyyyyyyy zzzzzzzzzz| A1234567D@yahoo.com |
| Jeanette Lim | A1234567E | xxxxxxxxxx yyyyyyyyyy zzzzzzzzzz| A1234567E@qq.com |

---

## SECTION 4 : VIDEO OF SYSTEM MODELLING & USE CASE DEMO

[![Sudoku AI Solver](http://img.youtube.com/vi/-AiYLUjP6o8/0.jpg)](https://youtu.be/-AiYLUjP6o8 "Sudoku AI Solver")

Note: It is not mandatory for every project member to appear in video presentation; Presentation by one project member is acceptable. 
More reference video presentations [here](https://telescopeuser.wordpress.com/2018/03/31/master-of-technology-solution-know-how-video-index-2/ "video presentations")

---

## SECTION 5 : USER GUIDE

`Refer to appendix <Installation & User Guide> in project report at Github Folder: ProjectReport`

### [ 1 ] To run the system using iss-vm

> download pre-built virtual machine from http://bit.ly/iss-vm

> start iss-vm

> open terminal in iss-vm

> $ git clone https://github.com/telescopeuser/Workshop-Project-Submission-Template.git

> $ source activate iss-env-py2

> (iss-env-py2) $ cd Workshop-Project-Submission-Template/SystemCode/clips

> (iss-env-py2) $ python app.py

> **Go to URL using web browser** http://0.0.0.0:5000 or http://127.0.0.1:5000

### [ 2 ] To run the system in other/local machine:
### Install additional necessary libraries. This application works in python 2 only.

> $ sudo apt-get install python-clips clips build-essential libssl-dev libffi-dev python-dev python-pip

> $ pip install pyclips flask flask-socketio eventlet simplejson pandas

---
## SECTION 6 : PROJECT REPORT / PAPER

`Refer to project report at Github Folder: ProjectReport`

**Recommended Sections for Project Report / Paper:**
- Executive Summary / Paper Abstract
- Sponsor Company Introduction (if applicable)
- Business Problem Background
- Market Research
- Project Objectives & Success Measurements
- Project Solution (To detail domain modelling & system design.)
- Project Implementation (To detail system development & testing approach.)
- Project Performance & Validation (To prove project objectives are met.)
- Project Conclusions: Findings & Recommendation
- Appendix of report: Project Proposal
- Appendix of report: Mapped System Functionalities against knowledge, techniques and skills of modular courses: MR, RS, CGS
- Appendix of report: Installation and User Guide
- Appendix of report: 1-2 pages individual project report per project member, including: Individual reflection of project journey: (1) personal contribution to group project (2) what learnt is most useful for you (3) how you can apply the knowledge and skills in other situations or your workplaces
- Appendix of report: List of Abbreviations (if applicable)
- Appendix of report: References (if applicable)

---
## SECTION 7 : MISCELLANEOUS

`Refer to Github Folder: Miscellaneous`

### HDB_BTO_SURVEY.xlsx
* Results of survey
* Insights derived, which were subsequently used in our system

---

### <<<<<<<<<<<<<<<<<<<< End of Template >>>>>>>>>>>>>>>>>>>>

---

**This [Machine Reasoning (MR)](https://www.iss.nus.edu.sg/executive-education/course/detail/machine-reasoning "Machine Reasoning") course is part of the Analytics and Intelligent Systems and Graduate Certificate in [Intelligent Reasoning Systems (IRS)](https://www.iss.nus.edu.sg/stackable-certificate-programmes/intelligent-systems "Intelligent Reasoning Systems") series offered by [NUS-ISS](https://www.iss.nus.edu.sg "Institute of Systems Science, National University of Singapore").**

**Lecturer: [GU Zhan (Sam)](https://www.iss.nus.edu.sg/about-us/staff/detail/201/GU%20Zhan "GU Zhan (Sam)")**

[![alt text](https://www.iss.nus.edu.sg/images/default-source/About-Us/7.6.1-teaching-staff/sam-website.tmb-.png "Let's check Sam' profile page")](https://www.iss.nus.edu.sg/about-us/staff/detail/201/GU%20Zhan)

**zhan.gu@nus.edu.sg**

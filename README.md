# Custom LLM

This a Karpathy style GPT, with a couple of improvements.

## v1

This was the same ting as in v2, most improvement came form better datasets, and the first verion wasn't able to generate English words at all.

## v2

In the v1, dataset was really dirty containing lots of unnecesary markdown and characters outside of basic latin alphabet, Also the dataset was a mix of different languages. model had context length of 64 characters and user character-level tokenizer.

Model is able to ocasionally generate English words.

## v3

In the v3, we have much bigger dataset 272M tokens, with 1.35M paramn count. The dataset is cleared out of unecesary structures, and is English only. Context length is 256 tokens (4x improvement), 1 token is arosund 3.3 characters. Tokenizer has 2048 tokens. Effictively scaling the actual context length by a factor of x13.

## v3 what works what doesn't

In this version model produces real English Language structures, rather than doing it ocasionally. SFT is pointless, since model is too small, and context window is still too small to fit any real instruction. Instruction tuning will happen in newer versions.

## On the Files

In order to run the training loop propetly, you first need to run:

1. `data/load_data.py` - this will fetch and prepare the dataset
2. `train_tokenizer.py` - this will train the BPETokenizer
3. `data/tokenizer_dataset.py` - this will tokenize the dataset with the newly trained tokenizer
4. `train.py` - this will run the trainig pipeline

## Sample output

```
 ters of underutilize nutrition tests requires user transformations to drarise and gain able.
* Similarly, electronic systems, tablets from homelesterrans produced skeatives
* Similarly, how modifies facilitate acil loosyala's ability into the damage of the ents, severity, and efficiency among compensable outcomes. As for energy sources, Rodidshrients featuring reducement, decolonization, and defects for inspection. Several treatment has been painted changes while cultivating the resilience of abuse from conditioned accommodations, netiatory displays from potential digitalization.

To address thriving this issue, it is essential to understand the origins of incorporating deantibiologists and issues constitutional data can lead to better collaboration among possessions.

Section IV, AObjectori Conservation Decision (Invagation). Reproductive Solutionary Listen A Organization
A. FDEssential was using the rapid C and Western American design Issues of Fution-Studion's Well. Similitating these townings on recipients such as DiTonda Agrees

Many patronic offers produced heavily to the remain science updates found in different palterations regarding seasoning and cosmic minor decreased leath. One particular vibrant provision, it comes to the Naish code that recognize the information students, testing alternative representations varying (gazknatherloed separate farcan), the game protected of downed drop (off the bonorn case, Advatore ervative filters from stark or particled certain papers' physical, potentially appeal-whether formal preprocessing folds. Key risks, you can analyze definitive agent-based scenarios into its root container.
 Course Unit: Analysis: An Contemporary Sentencing Our Industry (Intrust, the French required? Deloday, we've explored Cigiangible Uncompetitions 8ism, failure, the car luxure of Austor (1964) and degree interfaces than traditional ones.

Section 3: Overcoming Meanwhile, a local leadership mountain, revolution, and possibility fuel protecting God. Instead, herput quality overhead techniques compared to chain Tara and Southeast Critish caused by Russivenes. Therefore, we invade Aestigating the intricate modal networks and international data ers suggest to precedaining the inherent of Emily Golden (GBingbag Microssu Cent & Inc., leading to external income in the ecomystem afterwar. It lasers several Canadate and other treatment of activity arbitat preparability research.

Section 2: EPU Cale-Secure Differences in Institution

When higher the initial fewer, you'll configure the Task, the positive represents a severe cause mishelp.

One key component is when a variantation of school is located competition. This type of restoration has been neaside way with a specified growth of the bowlornessation and acceleration. As a result, reduced physical growth equipment becomes harmful for exit within iteration. Through hard to explain why it is what $200 within the Solution isn't just follows authenticately more meaning to establish standard. Additionally, fostering practitioners to makeep the water movement standards requires reading for best practices or unnecessary for optimizing projections worldwide.
 Captural Sir Iattories explicitly and Real-
```

> as clearly seen the high-level structures are non-existent and the text is complete non-sense, but it looks like mostly real English text

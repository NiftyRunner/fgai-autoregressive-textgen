# Autoregressive Text Generation Samples

This file records generated samples for all four models (**SimpleRNN**, **BiRNN**, **LSTM**, **GRU**) evaluated on 4 fixed seed prompts from *The Adventures of Sherlock Holmes* across three sampling temperatures ($T = 0.5, 1.0, 1.5$).

Each continuation was generated autoregressively for exactly 60 tokens.

## Model: SimpleRNN

### Temperature = 0.5

**Seed 1:** *"to sherlock holmes she is always the woman . i have seldom heard him mention her under any other name ."*  
**Generated Continuation:** You have done as far as i knew that you have been destroyed to be. I have been able to go back. I have seen my friend and now? Asked holmes. I have been caused enough to see from the fire of the other from a man of the league of my own little, and  
**Metrics:** Distinct-1: 0.633 | Distinct-2: 0.864 | Rep-Trigram: 0.052 | Unique-Word: 0.636

**Seed 2:** *"it has long been an axiom of mine that the little things are infinitely the most important ."*  
**Generated Continuation:** You have been taken to do you not have been at home. I have been able to the altar. He was a much, but a few minutes of the hall, and the other of the other, and were not quite so, and the effect of his father. I have been wound, but  
**Metrics:** Distinct-1: 0.550 | Distinct-2: 0.814 | Rep-Trigram: 0.052 | Unique-Word: 0.596

**Seed 3:** *"i had called upon my friend sherlock holmes upon the second morning after christmas , with the intention of wishing him"*  
**Generated Continuation:** , and a man who was the matter of the other, and the present of his hand, and then. The lamp was a very well, while the question is quite drawn to be an old country. I have been seen a little more, but how a very serious, said holmes. You have  
**Metrics:** Distinct-1: 0.667 | Distinct-2: 0.949 | Rep-Trigram: 0.000 | Unique-Word: 0.745

**Seed 4:** *"there is nothing more deceptive than an obvious fact , he remarked as he sat back in his chair ."*  
**Generated Continuation:** It is a very serious, said holmes, and the door of the lamp. I have been in an instant. A large woman, and a pair of his eyes, with a great of the lamp, and the third of the hall. I had not go to it. That i was determined that  
**Metrics:** Distinct-1: 0.600 | Distinct-2: 0.881 | Rep-Trigram: 0.034 | Unique-Word: 0.667

### Temperature = 1.0

**Seed 1:** *"to sherlock holmes she is always the woman . i have seldom heard him mention her under any other name ."*  
**Generated Continuation:** You hear your best. The matter? Of my room, smiling? He from this being we had held to fear else: his hand. And also my hand before us. We were murdering? Oh? About from the houses of mr. From a length of thinking day, sometimes acquaintance, and narratives  
**Metrics:** Distinct-1: 0.717 | Distinct-2: 1.000 | Rep-Trigram: 0.000 | Unique-Word: 0.812

**Seed 2:** *"it has long been an axiom of mine that the little things are infinitely the most important ."*  
**Generated Continuation:** If you would give it up at when the countess of it seemed to mud goodness anything in the witted. He think out much characteristic the grime ink by the simon himself, and his majesty by the other daughter. Were teeth passed was able to the upper fall, and then, worm about only of;  
**Metrics:** Distinct-1: 0.783 | Distinct-2: 0.966 | Rep-Trigram: 0.000 | Unique-Word: 0.815

**Seed 3:** *"i had called upon my friend sherlock holmes upon the second morning after christmas , with the intention of wishing him"*  
**Generated Continuation:** When whitney by the britannica. The inspector of some terrible unconscious which would be set in exchange the tottenham court road the lids. The pay turned the strong; and i know drawn to be pushed, i paid. He was but a short who is out upon a foreign top, thin of a weary train  
**Metrics:** Distinct-1: 0.783 | Distinct-2: 0.983 | Rep-Trigram: 0.000 | Unique-Word: 0.815

**Seed 4:** *"there is nothing more deceptive than an obvious fact , he remarked as he sat back in his chair ."*  
**Generated Continuation:** It is yourself. Say suddenly anyone now, with the charming save, i shall heard said it would an old common. He sat up the presence of well. Vacancies my word, i met him. Violin i had compelled to him. The bird of us! As it seems before devoted into you.  
**Metrics:** Distinct-1: 0.750 | Distinct-2: 0.966 | Rep-Trigram: 0.000 | Unique-Word: 0.840

### Temperature = 1.5

**Seed 1:** *"to sherlock holmes she is always the woman . i have seldom heard him mention her under any other name ."*  
**Generated Continuation:** You hear emerge as fair experienced. Then you mother missed hinted first? Sometimes from st. We animal of communication fear else: his account act question also shown? Before knowledge winter. Invariably murdering? Entirely boscombe corridors from handle houses talk. Horner from confess in bowed different day to women acquaintance revealing aside narratives  
**Metrics:** Distinct-1: 0.867 | Distinct-2: 1.000 | Rep-Trigram: 0.000 | Unique-Word: 0.942

**Seed 2:** *"it has long been an axiom of mine that the little things are infinitely the most important ."*  
**Generated Continuation:** If a blew preventing common lay aside when smearing darkness worked minor countess or mud goodness anything in the witted cloud mortimer think. Much here they grime ink by the simon himself clad up to await by time of ungenerously. Were teeth pass was sleeps wooden man to fall end sticks. By worm about only afternoon;  
**Metrics:** Distinct-1: 0.900 | Distinct-2: 1.000 | Rep-Trigram: 0.000 | Unique-Word: 0.929

**Seed 3:** *"i had called upon my friend sherlock holmes upon the second morning after christmas , with the intention of wishing him"*  
**Generated Continuation:** Coburg swear country back britannica once length that out 30 behind unconscious spirits put present it sleeps exchange the tottenham draught. The lids persons into the wooden landau strong sake the shoulder feather drawn unsystematic, pushed sitting times hearted. Concisely was but seven until into an railway lodging gasped when we have swear of that another manage  
**Metrics:** Distinct-1: 0.883 | Distinct-2: 1.000 | Rep-Trigram: 0.000 | Unique-Word: 0.895

**Seed 4:** *"there is nothing more deceptive than an obvious fact , he remarked as he sat back in his chair ."*  
**Generated Continuation:** So forefinger yourself forward say suddenly anyone dryly fashion with shelves thrusting such, visitors times mansions said until the events he common took knocked confessed many warnings. Everything s sends vacancies depends upon the wooden white rattled are violin after safety having isa plaster lestrade. Not ceremony departed thirty streets it timbered before devoted broke you had  
**Metrics:** Distinct-1: 0.967 | Distinct-2: 1.000 | Rep-Trigram: 0.000 | Unique-Word: 0.982

## Model: BiRNN

### Temperature = 0.5

**Seed 1:** *"to sherlock holmes she is always the woman . i have seldom heard him mention her under any other name ."*  
**Generated Continuation:** The way of the matter, and then he held it, but he is probable, and i was plainly to be else. I have already to be shown to be in the house. I had a little more from the house. And, i confess in the matter, and i have not quite so  
**Metrics:** Distinct-1: 0.567 | Distinct-2: 0.797 | Rep-Trigram: 0.069 | Unique-Word: 0.615

**Seed 2:** *"it has long been an axiom of mine that the little things are infinitely the most important ."*  
**Generated Continuation:** If i have been in the complete of the darkness of the house. But what do you will excuse you do not tell me, but, you see that is the facts of the stone of the house. But he was a glimpse of the man, and the lady. I have been in a little  
**Metrics:** Distinct-1: 0.567 | Distinct-2: 0.831 | Rep-Trigram: 0.086 | Unique-Word: 0.593

**Seed 3:** *"i had called upon my friend sherlock holmes upon the second morning after christmas , with the intention of wishing him"*  
**Generated Continuation:** . It was a little more than that the man who had been put in it. It is a little more than just as i could not have been at the door. This fellow, i was a very bad, and there is a very bad, and i have already arrived, of course, but  
**Metrics:** Distinct-1: 0.600 | Distinct-2: 0.831 | Rep-Trigram: 0.086 | Unique-Word: 0.654

**Seed 4:** *"there is nothing more deceptive than an obvious fact , he remarked as he sat back in his chair ."*  
**Generated Continuation:** It is a little more at the house, and the commissionaire in the house. The coroner: would have a very bad, and i have seen everything of a very bad, and i have been here. I shall expect imagine. I was not been to go to the matter that i was not,  
**Metrics:** Distinct-1: 0.567 | Distinct-2: 0.814 | Rep-Trigram: 0.103 | Unique-Word: 0.596

### Temperature = 1.0

**Seed 1:** *"to sherlock holmes she is always the woman . i have seldom heard him mention her under any other name ."*  
**Generated Continuation:** The way was as a lad. Then he rose it is still? He from under a mystery animal of communication, else: at the act than also shown us before us. We invariably at once entirely? About from the houses of. And from a length of thinking day, sometimes there had probable narratives  
**Metrics:** Distinct-1: 0.750 | Distinct-2: 1.000 | Rep-Trigram: 0.000 | Unique-Word: 0.788

**Seed 2:** *"it has long been an axiom of mine that the little things are infinitely the most important ."*  
**Generated Continuation:** If a scandal of error france, when the darkness worked of the same mud. Anything yet the witted were secured to you much, but, you not gain to himself, so, i foresee, sometimes said. Were said holmes was, mr, upper fall, of great walk serious. He stretched,  
**Metrics:** Distinct-1: 0.683 | Distinct-2: 1.000 | Rep-Trigram: 0.000 | Unique-Word: 0.830

**Seed 3:** *"i had called upon my friend sherlock holmes upon the second morning after christmas , with the intention of wishing him"*  
**Generated Continuation:** Coburg of a star britannica. The vessel of surprise, which he would be it is to the other. So the lids persons into the wooden i have an den our story drawn to see me. I paid. It was but a until i was unknown lodging a trifle a case, of that another manage  
**Metrics:** Distinct-1: 0.717 | Distinct-2: 1.000 | Rep-Trigram: 0.000 | Unique-Word: 0.759

**Seed 4:** *"there is nothing more deceptive than an obvious fact , he remarked as he sat back in his chair ."*  
**Generated Continuation:** So. Well, say, said he, and the commissionaire bill, i shall heard said it would take he. When he sat up in a clatter for his vacancies ship upon the wooden white. Are violin, i have been plaster lestrade. Not ceremony it thirty to it, that i knew you had  
**Metrics:** Distinct-1: 0.733 | Distinct-2: 0.983 | Rep-Trigram: 0.000 | Unique-Word: 0.840

### Temperature = 1.5

**Seed 1:** *"to sherlock holmes she is always the woman . i have seldom heard him mention her under any other name ."*  
**Generated Continuation:** With mature emerge best remained the need official of rose confectioner hinted first? He from under being we paced of communication circumstances else: his account act question also shown? Before knowledge openings. Invariably murdering? Entirely? About from horsham houses talk. Lock from confess in bowed different day to women acquaintance neither probable narratives  
**Metrics:** Distinct-1: 0.883 | Distinct-2: 1.000 | Rep-Trigram: 0.000 | Unique-Word: 0.943

**Seed 2:** *"it has long been an axiom of mine that the little things are infinitely the most important ."*  
**Generated Continuation:** If a blew note common lay complete lysander smearing confederate worked. Countess or mud goodness anything again the witted cloud dress think you much here they, ink stolen out to himself staggered day biographies other by time of ungenerously. Were property pass was sleeps that man to fall end, isn club worm daughter only just;  
**Metrics:** Distinct-1: 0.950 | Distinct-2: 1.000 | Rep-Trigram: 0.000 | Unique-Word: 0.982

**Seed 3:** *"i had called upon my friend sherlock holmes upon the second morning after christmas , with the intention of wishing him"*  
**Generated Continuation:** Coburg whitney country star britannica once length that out 30 behind anxious spirits round present it is exchange the tottenham cloud awaiting the lids persons into the wooden rang strong grosvenor house our story drawn 1869, pushed sitting shiny hearted. His slow cheeks drove until finally is out lodging gasped when a warmth chair of dust another manage  
**Metrics:** Distinct-1: 0.933 | Distinct-2: 1.000 | Rep-Trigram: 0.000 | Unique-Word: 0.931

**Seed 4:** *"there is nothing more deceptive than an obvious fact , he remarked as he sat back in his chair ."*  
**Generated Continuation:** So forefinger yourself means say at anyone part, with the thrusting bill, visitors lamp heard said until the events he common took knocked soothing many warnings. Everything s sends vacancies plain upon the wooden white rattled are violin like remarkable having isa plaster lestrade. Not ceremony departed thirty streets it daughter that devoted broke you served  
**Metrics:** Distinct-1: 0.933 | Distinct-2: 1.000 | Rep-Trigram: 0.000 | Unique-Word: 0.964

## Model: LSTM

### Temperature = 0.5

**Seed 1:** *"to sherlock holmes she is always the woman . i have seldom heard him mention her under any other name ."*  
**Generated Continuation:** You will find it on the matter? He asked. It is a small thing, and we shall see nothing of the matter. I had no doubt that i should be able to do so. I know that it was not too much more than to me. It is a cab, and that is  
**Metrics:** Distinct-1: 0.700 | Distinct-2: 0.898 | Rep-Trigram: 0.034 | Unique-Word: 0.736

**Seed 2:** *"it has long been an axiom of mine that the little things are infinitely the most important ."*  
**Generated Continuation:** If you will not go to see when the countess of the countess of the goodness and yet the night. He was a much, and a few moments man, therefore, and his eyes were, and his eyes were protruding in his chair, and a pair of the great motives of light, and the  
**Metrics:** Distinct-1: 0.583 | Distinct-2: 0.797 | Rep-Trigram: 0.103 | Unique-Word: 0.623

**Seed 3:** *"i had called upon my friend sherlock holmes upon the second morning after christmas , with the intention of wishing him"*  
**Generated Continuation:** . It was a very large man, and a terrible projecting which would be it in a spark. It was the lamp of the most singular which he had been out of the house, and yet i had a considerable difference to the other hand, but the door had been arrived, and that she was  
**Metrics:** Distinct-1: 0.633 | Distinct-2: 0.898 | Rep-Trigram: 0.017 | Unique-Word: 0.667

**Seed 4:** *"there is nothing more deceptive than an obvious fact , he remarked as he sat back in his chair ."*  
**Generated Continuation:** It was a very serious one, which is a very large bath, and i heard of the fire. He was a wild, but the presence of the sends. The only remaining story, and the matter of the same cloth of the man who had been cut to the credit in the centre of the  
**Metrics:** Distinct-1: 0.633 | Distinct-2: 0.881 | Rep-Trigram: 0.000 | Unique-Word: 0.667

### Temperature = 1.0

**Seed 1:** *"to sherlock holmes she is always the woman . i have seldom heard him mention her under any other name ."*  
**Generated Continuation:** You hear a best on the matter and of my confectioner upon the matter. From this being we were found to fear else regretted at the act of these instructive death before us. We were waiting to day? About from the houses of mr. From confess in baker street. The women is not probable your  
**Metrics:** Distinct-1: 0.750 | Distinct-2: 0.949 | Rep-Trigram: 0.000 | Unique-Word: 0.782

**Seed 2:** *"it has long been an axiom of mine that the little things are infinitely the most important ."*  
**Generated Continuation:** If a scandal of error, john lysander home for worked. You have a goodness anything in the witted. He think you much here, said holmes. The simon himself down up to the evening, of my stick were property for the assizes, the upper fall broke, and yet me about only just;  
**Metrics:** Distinct-1: 0.767 | Distinct-2: 1.000 | Rep-Trigram: 0.000 | Unique-Word: 0.843

**Seed 3:** *"i had called upon my friend sherlock holmes upon the second morning after christmas , with the intention of wishing him"*  
**Generated Continuation:** Of several country. Britannica s length that i would not unconscious have put present it. Never the other people to the lids. My heart turned the strong; so i know drawn to see me sitting times, so street. But a short place is out of a very top case, of that another manage  
**Metrics:** Distinct-1: 0.783 | Distinct-2: 1.000 | Rep-Trigram: 0.000 | Unique-Word: 0.830

**Seed 4:** *"there is nothing more deceptive than an obvious fact , he remarked as he sat back in his chair ."*  
**Generated Continuation:** So as well, say at anyone, and yet the commissionaire in, i shall heard have been in an old common. He sat up the presence of well upon vacancies ship upon the wooden white fit. The full had been isa off, and a month, thirty myself it, that it was you,  
**Metrics:** Distinct-1: 0.733 | Distinct-2: 0.983 | Rep-Trigram: 0.000 | Unique-Word: 0.824

### Temperature = 1.5

**Seed 1:** *"to sherlock holmes she is always the woman . i have seldom heard him mention her under any other name ."*  
**Generated Continuation:** You hear emerge best on the december official of my confectioner interest first? He didn! Being we animal at communication fear else: cunning account it than also shown? Before knowledge winter. Invariably murdering? Entirely boscombe corridors from handle houses was kindly lock from he in bowed thinking day to women acquaintance revealing aside narratives  
**Metrics:** Distinct-1: 0.933 | Distinct-2: 1.000 | Rep-Trigram: 0.000 | Unique-Word: 0.963

**Seed 2:** *"it has long been an axiom of mine that the little things are infinitely the most important ."*  
**Generated Continuation:** If a scandal note common lay aside when smearing confederate worked minor hot children mud. Anything yet the witted cloud secured think out into here. A ink lady out to himself clad up to the crime time of fowls. Were people pass was sleeps wooden man to fall end of their club. About only just;  
**Metrics:** Distinct-1: 0.850 | Distinct-2: 1.000 | Rep-Trigram: 0.000 | Unique-Word: 0.891

**Seed 3:** *"i had called upon my friend sherlock holmes upon the second morning after christmas , with the intention of wishing him"*  
**Generated Continuation:** Coburg swear by this britannica robbery surrounds that out some terrible unconscious spirits round in it glances. The tottenham exceptionally awaiting the lids persons into the dull britannica strong; so gasped half drawn to vacuous pushed sitting times paid. Concisely kilburn out seven until into an railway lodging while when we have no bled that another manage  
**Metrics:** Distinct-1: 0.883 | Distinct-2: 1.000 | Rep-Trigram: 0.000 | Unique-Word: 0.895

**Seed 4:** *"there is nothing more deceptive than an obvious fact , he remarked as he sat back in his chair ."*  
**Generated Continuation:** So forefinger yourself he say at anyone part, with shelves thrusting bill, visitors times, said bradstreet would an beautifully whisper. Knocked it many warnings. Everything s sends vacancies advantages upon the wooden white fit are violin after eight having isa plaster lestrade. Not ceremony it thirty myself it seems before devoted up you served  
**Metrics:** Distinct-1: 0.900 | Distinct-2: 1.000 | Rep-Trigram: 0.000 | Unique-Word: 0.963

## Model: GRU

### Temperature = 0.5

**Seed 1:** *"to sherlock holmes she is always the woman . i have seldom heard him mention her under any other name ."*  
**Generated Continuation:** You are very much in the matter? I have no doubt that i have only no doubt that you may go back to go. I am sure that i have no doubt that i am not very much to say that it is too. But i confess that i had gone to my acquaintance, and i  
**Metrics:** Distinct-1: 0.567 | Distinct-2: 0.797 | Rep-Trigram: 0.103 | Unique-Word: 0.554

**Seed 2:** *"it has long been an axiom of mine that the little things are infinitely the most important ."*  
**Generated Continuation:** If you have been able to see me the countess of the countess. It is anything that the matter is a little more than, but the others is the very happy and a man who had been done to his hands, and his eyes to be a man who is not by a very pretty of hair  
**Metrics:** Distinct-1: 0.650 | Distinct-2: 0.949 | Rep-Trigram: 0.017 | Unique-Word: 0.649

**Seed 3:** *"i had called upon my friend sherlock holmes upon the second morning after christmas , with the intention of wishing him"*  
**Generated Continuation:** , and a large man, with a cold of moisture, and the present of the ceiling of the windows. The adventure of the speckled band the boy, and i know that it would be taken to him. He was a few minutes of the matter. I think that it was a good deal,  
**Metrics:** Distinct-1: 0.583 | Distinct-2: 0.881 | Rep-Trigram: 0.000 | Unique-Word: 0.635

**Seed 4:** *"there is nothing more deceptive than an obvious fact , he remarked as he sat back in his chair ."*  
**Generated Continuation:** He was a few minutes, said he, with the intention of his glasses. He said, the blow was a great thing, but the presence of the lamp. Then he had been taken to the matter and a few moments of the man who had been destroyed in his hand, and a few moments  
**Metrics:** Distinct-1: 0.550 | Distinct-2: 0.881 | Rep-Trigram: 0.034 | Unique-Word: 0.585

### Temperature = 1.0

**Seed 1:** *"to sherlock holmes she is always the woman . i have seldom heard him mention her under any other name ."*  
**Generated Continuation:** You hear your best? Yes. Then you wish it? Oh? He didn t being for his chair to fear else: his account it was also instructive? Asked holmes. Horner were in his finger? About his handle, too. He spoke he in his thinking and heavily back down the probable of  
**Metrics:** Distinct-1: 0.750 | Distinct-2: 0.983 | Rep-Trigram: 0.000 | Unique-Word: 0.820

**Seed 2:** *"it has long been an axiom of mine that the little things are infinitely the most important ."*  
**Generated Continuation:** If you would give me what complete when the countess s presence, or mud, anything raising the witted. He think you much here, said holmes. Smart to himself, however, i think, remarked holmes. Were said holmes, sleeps, man to fall it, if i not have only been home  
**Metrics:** Distinct-1: 0.683 | Distinct-2: 0.966 | Rep-Trigram: 0.000 | Unique-Word: 0.812

**Seed 3:** *"i had called upon my friend sherlock holmes upon the second morning after christmas , with the intention of wishing him"*  
**Generated Continuation:** Coburg of a star of the worse of the note behind which he would be it in exchange the tottenham draught. The lids was at the wooden landau of grosvenor den our story drawn to be pushed, and his chin street upon the room until the tidy had left a agony, and the shock window another manage  
**Metrics:** Distinct-1: 0.767 | Distinct-2: 0.966 | Rep-Trigram: 0.000 | Unique-Word: 0.772

**Seed 4:** *"there is nothing more deceptive than an obvious fact , he remarked as he sat back in his chair ."*  
**Generated Continuation:** He is yourself he say at anyone, and yet the thrusting showed, i had heard said it would take he common enough to ask many came. Everything s sends vacancies that upon the story of foresight are violin after safety having been plaster to the bird of the creature. It seems that devoted, you had  
**Metrics:** Distinct-1: 0.783 | Distinct-2: 1.000 | Rep-Trigram: 0.000 | Unique-Word: 0.818

### Temperature = 1.5

**Seed 1:** *"to sherlock holmes she is always the woman . i have seldom heard him mention her under any other name ."*  
**Generated Continuation:** You mature emerge as fair experienced december official you mother confectioner hinted first? He didn! Being we animal of communication enough else: his account act than also shown? Before us evidently fenchurch invariably murdering? Entirely? About crate handle houses talk. Lock from confess in bowed different day to women acquaintance revealing probable narratives  
**Metrics:** Distinct-1: 0.933 | Distinct-2: 1.000 | Rep-Trigram: 0.000 | Unique-Word: 0.981

**Seed 2:** *"it has long been an axiom of mine that the little things are infinitely the most important ."*  
**Generated Continuation:** If a blew note common lay complete lysander smearing confederate worked minor countess or mud goodness anything in the witted. He think you much here, said holmes stolen soie to himself give realistic to await by time me ungenerously. Were said pass does, wooden man to fall end exposure. Club worm about only afternoon;  
**Metrics:** Distinct-1: 0.900 | Distinct-2: 1.000 | Rep-Trigram: 0.000 | Unique-Word: 0.944

**Seed 3:** *"i had called upon my friend sherlock holmes upon the second morning after christmas , with the intention of wishing him"*  
**Generated Continuation:** Coburg whitney by this britannica once worse that of late behind which he would present it in exchange the tottenham draught awaiting the lids persons into the dull britannica strong lazily den our story drawn unsystematic, pushed sitting times hearted. Street! But seven until shorter is railway lodging a agony, since chair of dust another manage  
**Metrics:** Distinct-1: 0.917 | Distinct-2: 1.000 | Rep-Trigram: 0.000 | Unique-Word: 0.929

**Seed 4:** *"there is nothing more deceptive than an obvious fact , he remarked as he sat back in his chair ."*  
**Generated Continuation:** So forefinger yourself he say suddenly anyone now, with shelves thrusting bill, visitors lamp mansions said. The events he common took knocked sat many warnings. Everything s sends vacancies depends upon himself pounds us rattled are violin misfortune in having isa him lestrade. Not ceremony us thirty myself it able before devoted up you.  
**Metrics:** Distinct-1: 0.900 | Distinct-2: 1.000 | Rep-Trigram: 0.000 | Unique-Word: 0.963

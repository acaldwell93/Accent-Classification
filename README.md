# Capstone 3 Proposals

## 1. Art Restoration with Autoencoders

I don't know if this falls under the realm of biting off more than I can chew, but I'd like to try training a GAN to restore faded or damaged artwork. I'm open to focusing on whatever works might be most feasable, but at the moment I'm most interested in focusing on ancient Roman mosaics, like the ones found here: https://www.metmuseum.org/blogs/now-at-the-met/features/2010/the-roman-mosaic-from-lod-israel https://www.metmuseum.org/art/collection/search/253565 https://www.metmuseum.org/art/collection/search/254535. (I'd predominantly used the Met API supplemented with any other museum resources I can to collect the data) Again, this may not be within the scope of the capstone, so I'd be open to any suggestions as to how I might narrow the focus or change the art subject matter to something more reasonable.

## 2. UK accent classifier

If you've ever watched British television, you'll know the UK is a cornucopia of different accents. To those with untrained ears outside of the UK, these accents might a little difficult to parse, and perhaps even harder to fully identify. But in many of these shows, predominantly intended for a British audience, the audience might be expected to hear an accent, identify it, and thus gain important information about a character. Essentially, accent identification can be an aspect of the storytelling. So the idea for this project would be to develop a kind of Shazam for British accents, allowing a user to play audio of a speaking character and identify the most likely region of origin for a character. Would use this dataset (https://research.google/tools/datasets/uk-ireland-english-dialects/), and likely would stick with its fairly broad 6 classes/regions in the UK and Ireland.

## 3. Political Spectrum Similarity / Recommender System for National Politics

Probably focusing on House voting records from the last full Congress (116th Congress, 2019-2021), the idea here would be to first use clustering techniques to identify political groups/factions based solely on voting record. Then, using dimensionality reduction to identify the most salient features and combining that with user input in a recommender system that could help them compare their own political opinions with those of their own House representives or other prominent members. 

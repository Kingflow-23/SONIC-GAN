# TOAD-GAN: Procedural Generator of 2D Platform Game Levels

## Table of Contents

- [a. Introduction](#a-introduction)  
  - [i. Context](#i-context)  
  - [ii. State of the art](#ii-state-of-the-art)  
  - [iii. Contribution](#iii-contribution)  
  - [iv. Resources](#iv-resources)  
- [b. Work Done](#b-work-done)  
  - [i. Research and Preparation](#i-research-and-preparation)  
  - [ii. Dataset Collection and Preparation](#ii-dataset-collection-and-preparation)  
  - [iii. Model Adaptation & Level Generation](#iii-model-adaptation--level-generation)  
  - [iv. Integration into Unity](#iv-integration-into-unity)  
  - [v. Evaluation](#v-evaluation)  
- [c. Results](#c-results)  
- [d. Conclusion](#d-conclusion)  
- [e. Skills Acquired](#e-skills-acquired)  
  - [Hard Skills](#hard-skills)  
  - [Soft Skills](#soft-skills)  
- [f. Appendix](#f-appendix)  
- [g. References](#g-references)

---

## a. Introduction

Procedural Content Generation (PCG) has become a pivotal technique in modern game development, enabling the automated creation of game levels, assets, and mechanics. It enhances the player's experience by providing variety, replayability, and scalability while reducing development time. With the advancements in Artificial Intelligence (AI), deep learning techniques such as Generative Adversarial Networks (GANs) have been increasingly integrated into PCG, allowing for more dynamic and sophisticated content generation.

Building upon our previous work with TOAD-GAN for 2D platform game level generation, this project aims to further refine and expand the capabilities of AI-driven procedural generation. In this new work, we seek to enhance the model's adaptability, improve level design complexity, and seamlessly integrate the generated levels into Unity for a more immersive gaming experience.

Our primary objective is to optimize the procedural generation pipeline, ensuring that generated levels are not only visually appealing but also mechanically sound and engaging. By refining the dataset, adjusting hyperparameters, and incorporating new evaluation metrics, we will work toward producing high-quality, fully functional Sonic-inspired levels.

---

### i. Context

#### Previous Work and Motivation

Our previous project successfully demonstrated the feasibility of using TOAD-GAN to generate 2D platformer levels inspired by Sonic the Hedgehog. The generated levels maintained structural integrity, aesthetics, and playability but required further improvements in terms of difficulty balancing, enemy placement, and overall gameplay dynamics. Additionally, the integration into Unity was in its early stages, highlighting the need for further development to ensure seamless gameplay experiences.

#### Project Goals for This Semester

In this work, we aim to build upon our prior work by focusing on the following key objectives:

1. **Refining TOAD-GAN's Performance**
   - Utilizing commercially designed levels as benchmarks to achieve realistic level design.
   - Enhancing the dataset to include more diverse level structures and gameplay elements.
   - Fine-tuning hyperparameters to improve the quality and variability of generated levels.
   - Addressing issues such as mode collapse and ensuring better difficulty progression.
2. **Advancing Unity Integration**
   - Developing a more structured pipeline to import GAN-generated levels into Unity.
   - Implementing logic and physics adjustments to improve level navigation and playability.
   - Introducing AI-driven playtesting to evaluate level design effectiveness.
3. **Implementing New Evaluation Metrics**
   - Establishing quantitative and qualitative measures for level assessment.
   - Using AI agents to test level playability and difficulty scaling.
   - Comparing GAN-generated levels with human-designed levels for benchmarking.

#### Expected Outcomes

By the end of this project, we aim to:

- Produce high-quality, procedurally generated Sonic-inspired levels with improved complexity and design.
- Achieve full integration with Unity, ensuring that generated levels can be seamlessly played and tested.
- Establish a robust evaluation framework for assessing the effectiveness of generated levels.
- Lay the groundwork for potential expansions, including multiplayer compatibility or additional game mechanics.

This continued research and development will not only refine our GAN-based approach but also contribute valuable insights to the broader field of AI-driven game design.

*In the following sections, we detail our methodology, experimental setup, results, and future work.*

---

### ii. State of the art

*(Add relevant content regarding current approaches and literature here.)*

---

### iii. Contribution

*(Detail your specific contributions to the field, improvements over previous work, and any novel techniques you introduced.)*

---

### iv. Resources

*(List any external resources, libraries, datasets, or tools used in the project.)*

---

## b. Work Done

### i. Research and Preparation

To advance the project, we extended our research by incorporating commercial Sonic levels and exploring broader level map sheets. The objective was to ensure that our dataset covered a more extensive variety of existing Sonic levels, providing a more comprehensive foundation for our GAN model.

- **Sprite Collection:**  
  Unlike the previous approach, where we primarily relied on publicly available repositories and fan communities, we now source commercial-level data directly from Sonic games. Analyzing commercial Sonic levels provided access to richer and more intricate structures, which in turn refined our procedural generation process.  
  We expanded our dataset with broader level map sheets, incorporating multiple environments, platforms, hazards, and background assets representing different stages of Sonic games. This broader scope enhances the model's ability to generate levels with greater structural variety and gameplay depth.

  *Examples of newly sourced assets are shown in the following figures:*
  - **Fig 1:** Commercial Sonic level spritesheet  
  - **Fig 2:** Extended Mapsheet  
  - **Fig 3:** Previously retained sprites such as Sonic, goalposts, and enemies.

- **Sprite Adaptation:**  
  To ensure compatibility with the TOAD-GAN model, many of these commercial assets required preprocessing and adaptation. Unlike the static sprites used in the previous work, the new dataset includes more dynamic elements such as moving platforms and interactive traps.  
  Additionally, we restructured our token hierarchy to reflect the complexity of commercial Sonic levels, refining the token categorization system to accommodate a wider range of terrain features, obstacles, and interactive elements. This adaptation is crucial for improving the model’s ability to generate more complex and engaging levels.

  By incorporating commercial-level maps and refining our sprite dataset, we have significantly enhanced our research foundation, setting the stage for improved procedural level generation in the subsequent phases of our project.

---

### ii. Dataset Collection and Preparation / ASCII Auto-Encoder

For this semester, we shifted our approach to dataset collection by moving away from AI-generated ASCII levels and focusing on commercial Sonic levels. This decision was made to better capture the complexities of Sonic’s level design and integrate more authentic game data into our procedural generation pipeline.

#### GenSonic

GenSonic (version 0) is the tool we developed to automatically convert the image of a level—specifically, the first level of Sonic the Hedgehog—into its ASCII representation. We built this tool because, with the GAN now tailored to a commercial level, we need to provide the GAN with an ASCII-encoded level. However, the level is simply too large to be encoded manually within a reasonable timeframe.

GenSonic operates in four phases:

1. **The 256-Split**  
   The first step involves splitting the input image into 256 blocks, as the level is an assembly of 256×256 image blocks arranged sequentially to form the complete level.  
   **Fig 5:** Commercial level to be divided into 64×64 tiles.

2. **The 64-Split**  
   GenSonic processes the level image along with the sprite pool, which is the set of 64-pixel images that compose each 256-block of the original level.  
   **Fig 6:** ASCII Matrix of a 256-block

3. **Sprite Matching and ASCII Conversion**  
   Once the level is split, we use an engine that employs two dictionaries. The first maps each image from the sprite pool to its name (with the sprite name as the key and the image array as the value). The second maps each sprite name to an ASCII character. Since there are more sprites than available ASCII characters, we build the mapper of sprite names to ASCII characters based only on the sprites present in the level.  
   For each 64-sprite within each block, we compare it to every image in the sprite pool and replace it with the closest match. Using the resulting 64-sprite’s name, we retrieve its associated ASCII character. This process produces an ASCII transcription matrix for each reconstructed 256-block.

4. **Final Reconstruction**  
   Finally, we combine all the ASCII character matrices from each block to recreate the full ASCII matrix of the original level.  
   **Fig 7:** ASCII Matrix of the full level

This is the first version of GenSonic. Although there is significant room for improvement and optimization, it achieved the intended task and saved us enough time to begin implementing the AI in Unity.

For future improvements in version v0.1, we plan to generalize GenSonic for other levels while maintaining high encoding accuracy. Although GenSonic can encode any Sonic level, it is currently most accurate for the first level of Sonic since the encoder was specifically trained on that level’s data.

- **Main Color Extraction and Sprite Matching:**  
  Instead of relying on manually designed levels, we source commercial-level data directly from Sonic games. The preprocessing would begin by extracting the dominant color from the entry level of each commercial Sonic level. This color would serve as a key reference point to help automate the sprite matching process. By associating the extracted color with corresponding sprite sheets found online, we streamline the assignment of the correct tiles during level encoding.

- **Automated Level Encoding:**  
  Once the main color is extracted, the system automatically selects the appropriate sprites based on this color reference. This automated method eliminates the need for manual sprite selection, reduces time spent on data preparation, and ensures consistency and accuracy in sprite matching, allowing for smoother procedural generation. This approach enhances the model’s ability to create varied levels—from backgrounds to obstacles—that are more faithful to the source material.

  **Fig 4:** Example of color matching for our commercial level. Closest sprites found refer to the Level 1 sprite set seen in Fig 2.

Although we did not have time to fully adapt GenSonic to the color detection engine, the code is already in place and can be completed in the next upgrade. By automating the encoding and sprite matching process, this semester’s dataset preparation became significantly more efficient. This new approach allows us to generate levels from any commercial Sonic level with ease, greatly improving the scalability of our procedural generation system.

---

### iii. Model Adaptation & Level Generation

This semester, we made adjustments to the TOAD-GAN model to improve its ability to handle the new dataset, which now consists of commercial Sonic levels. To simplify the task, we treated the commercial Sonic level as a standalone game. With the knowledge acquired last semester on how the GAN functions, adapting it to a commercial level was straightforward. The main challenge was building the dataset—the ASCII level that will be provided to the GAN.  
These changes aimed to enhance the model's adaptability to different level structures, ensure better integration with the automated dataset preparation pipeline, and increase the complexity of the generated levels. Instead of modifying the previously working model, we developed a completely new adaptation of the GAN to handle both “casual” Sonic and commercial levels.

**Fig 4:** Example of an ASCII level generated by the GAN

---

### iv. Integration into Unity

The project also aimed to design an AI capable of playing and finishing a GAN-created level to automate the testing after level generation. This phase was divided into three major parts:

#### a. Dynamic Level Instantiation

We built an environment that allows for easy testing of GAN-generated levels. Since the GAN output is a text file, we created a game object (the LevelLoader) that instantiates the level based on the input text file containing the level data. The LevelLoader has a script component where the prefab for each level block is assigned. The script scans the text file and, for each ASCII character, instantiates the corresponding prefab at the appropriate position.

**Fig 5:** Level Loader Structure

This structure enables us to automatically instantiate any level, provided we have the prefab and a dictionary mapping the prefab to the corresponding ASCII character. The implementation also includes the design of rules and physics for the prefabs, as well as game rules tailored to fit all the levels we want to test.

**Fig 6:** Graphical Rendering of the Commercial Level  
**Fig 7:** Original Commercial Level

#### b. Sonic Game Implementation

After dynamically instantiating levels in Unity, we recreated the Sonic game within Unity so that map changes would not affect game performance. This step was challenging, given our limited experience with Unity and C#. The final implementation resembles Sonic but is missing some mechanics, such as the rolling attack and rolling acceleration. However, it provided a functional platform for training an AI.

**Fig 8:** Simple Implementation of Sonic Game in Unity

In this basic game, a score counter tracks the player’s progress based on distance traveled, rings collected, and enemies defeated. Additionally, the player can perform multiple jumps, and enemy bullets have the capability to track the player.

#### c. AI Implementation

The final phase involved developing an AI to complete and test the levels. We chose a Reinforcement Learning approach because of our familiarity with its concepts and due to time constraints.  
Reinforcement Learning enables an agent to act in an environment by performing actions and receiving rewards or penalties, with the goal of maximizing cumulative rewards. This framework is ideal for dynamic, interactive tasks such as video game playing.

We used Unity ML Agents to configure our AI agent. The environment is the currently running level, and the objective for the agent is to finish the level. Rewards are provided for moving forward, collecting rings, and defeating enemies, while penalties are incurred for moving backward, taking damage, or losing rings.

**Fig 8:** Structure of the AI Agent

The AI training is executed via a terminal where the agent continuously observes the environment, performs actions, receives feedback, and updates its policy until the maximum number of epochs is reached.

---

### v. Evaluation

The evaluation phase focuses on analyzing the results of the AI in solving the level by monitoring the evolution of hyperparameters. Since the project is not yet complete—the AI is currently unable to finish the level—we cannot use metrics like time to completion or final score (which could be influenced by collectibles and enemy interactions) directly. Instead, we deduce the AI's strategy: whether it chooses not to collect all rings or defeat all enemies, or if it prioritizes reaching the end of the level as fast as possible. We also observe whether the AI's strategy changes based on the overall level design.

---

## c. Results

This project has led to the creation of several promising tools for game developers specializing in 2D platformer games. Although these tools are currently in their initial versions, they have the potential to significantly speed up the development of platformers.

- **GenSonic v0.0**  
  GenSonic is the ASCII auto-encoder that produces an ASCII text file corresponding to the input level. While its performance is not yet optimal, future improvements will make this tool very useful for game developers. When combined with the other tools, it will accelerate the overall process of level creation and testing.  
  *The tool can be found in its GitHub repository: [GenSonic](#)*

- **SONIC-GAN**  
  Working with GANs for six months deepened our understanding of their functionality, enabling us to adapt the model quickly and efficiently to various 2D platformer games. The GAN was successfully adapted to a Sonic-like game, and with a well-structured ASCII text file as input, it can generate up to 50 ASCII text files of level designs. These outputs can be used for testing or to dynamically generate levels during gameplay.  
  *The tool can be found in its GitHub repository: [SONIC-GAN](#)*

- **SONIC-UNITY**  
  SONIC-UNITY is a dynamic environment in Unity originally created for a Sonic game. It can also serve as a base for other 2D games and is designed for automatic testing using AI by recreating the game environment. The system uses a text file containing ASCII characters and accompanying dictionaries to instantiate the level. The only requirements are specifying the level text file and assigning the corresponding prefabs.  
  *The Unity environment can be found in its GitHub repository: [SONIC-UNITY](#)*

- **SONIC-AI**  
  SONIC-AI is designed to assess the levels generated by SONIC-GAN, thereby speeding up the level testing process. It uses reinforcement learning within the environment instantiated by SONIC-UNITY. Some AI versions have learned that advancing forward is the primary mechanic, increasing their score by moving through the level, though this is a basic strategy given that additional mechanics (such as jumping, attacking, and item collection) are also present.  
  The evolution of the AI's hyperparameters helps us determine which learning parameters to adjust or whether the environment itself needs modification.

  **Fig 9:** Evolution of the Epsilon Parameter  
  The epsilon (ε) parameter controls the epsilon-greedy strategy, balancing exploration and exploitation. A very low epsilon indicates that the AI does not explore new strategies. Increasing exploration could help the AI discover more effective ways to complete the level.

  **Fig 10:** Evolution of the Entropy Parameter  
  Similarly, the entropy parameter reveals the degree of unpredictability in the agent's actions, with low entropy indicating limited exploration.

  **Fig 11:** Evolution of the EVE Parameter  
  The Extrinsic Value Estimate (EVE) parameter reflects the agent’s prediction of future external rewards, i.e., those derived directly from the environment. A constant EVE value suggests that the AI might be stuck in a part of the environment where all actions yield similar or no rewards. This issue was observed when the AI became trapped in front of a ground block because it was unable to jump.

  *All trained AI versions (in ONNX format) can be found in the GitHub repository for the Unity portion of the project: [SONIC-UNITY](#)*

---

## d. Conclusion

In summary, this project has advanced the field of AI-driven procedural content generation by refining our previous work with TOAD-GAN. By transitioning from AI-generated levels to commercially sourced Sonic levels, we enhanced the authenticity and complexity of our dataset. Significant model adaptations—including a restructured token hierarchy, improved latent space representation, and targeted fine-tuning—have enabled the generation of levels that are both visually intricate and mechanically robust with respect to the input level. These developments not only strengthen the foundation for further integration into Unity but also explore the possibilities of AI in game development, especially for testing and level design. Moreover, this work paves the way for future expansions, such as enhanced multiplayer compatibility and additional game mechanics. Overall, our efforts contribute valuable insights into the capabilities and challenges of AI-driven game design.

---

## e. Skills Acquired

### Hard Skills
- **Procedural Content Generation (PCG):** Enhanced techniques for automated level design and asset integration.  
- **Deep Learning and GANs:** Advanced understanding and practical application of GAN architectures (specifically TOAD-GAN) for game level generation.  
- **Dataset Preparation:** Proficiency in sourcing and processing commercial game data, including color extraction, sprite matching, and tile cropping.  
- **Model Adaptation and Fine-Tuning:** Experience in restructuring neural network architectures to accommodate complex datasets and improve output variability.  
- **Unity Integration:** Skills in integrating AI-generated levels into the Unity engine, including handling game physics, level navigation, and playtesting using reinforcement learning.  
- **Preprocessing Techniques:** Implementation of automated methods for sprite encoding and standardization of dynamic game elements.

### Soft Skills
- **Critical Thinking and Problem Solving:** Ability to analyze complex challenges in level generation and devise effective technical solutions.  
- **Project Management:** Organized approach to research, development, and integration phases, ensuring timely progress and efficient resource utilization.  
- **Collaboration and Communication:** Effective teamwork and clear communication of technical concepts and project updates to both technical and non-technical stakeholders.  
- **Adaptability:** Flexibility in shifting methodologies (e.g., transitioning from AI-generated to commercially sourced levels) to better meet project objectives.  
- **Attention to Detail:** Meticulous handling of data preprocessing and model adjustments to ensure high-quality outputs.

---

## f. Appendix

1. **All Levels Mapsheet:** [Sonic the Hedgehog 3 Master System Tilesets](https://www.deviantart.com/pixelmarioxp/art/Sonic-the-Hedgehog-3-Master-System-Tilesets-863150342)  
2. **Commercial Sonic Level:** [ArtStation - Commercial Sonic Level](https://www.artstation.com/artwork/elZRoD)

---
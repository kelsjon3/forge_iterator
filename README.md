# Forge Iterator

An extension for Stable Diffusion WebUI (Forge/A1111) that allows you to cleanly iterate over a selected subfolder of checkpoints, generating a preset quantity of images per checkpoint, while perfectly preserving your prompt (including wildcards) and maintaining full compatibility with other Main scripts (like "One Button Prompt" or XYZ Plot).

## Features
- **AlwaysVisible Script**: Operates as an accordion under `txt2img` and `img2img` tabs.
- **Main Script Compatible**: Since it hooks into generation batches (`process_batch`) rather than taking over the entire `run` loop, it works flawlessly alongside your favorite Main scripts.
- **Dynamic Checkpoint Reloading**: Forces checkpoint swaps entirely on the fly during generation, ensuring tools relying on `alwayson_scripts` execute their states accurately.
- **Folder Filtering**: Automatically detects subdirectories in your `models/Stable-diffusion` path and organizes them into a drop-down. 
- **Iteration Multiplier**: Set the "Iterations (Batches) per Checkpoint" to 1+ across all models in a specific subdirectory, multiplying your base Batch Count accordingly.

## Installation
1. Navigate to your Stable Diffusion WebUI (Forge or Auto1111) `extensions/` directory.
2. `git clone https://github.com/kelsjon3/forge_iterator.git`
3. Restart or fully reload your WebUI.

## Usage
1. Open the *txt2img* or *img2img* tab.
2. Scroll to the "Forge Iterator" accordion at the bottom.
3. Check **Enable Forge Iterator**.
4. Select the **Checkpoint Subfolder** containing your desired models.
5. Set the **Iterations (Batches) per Checkpoint** slider. 
   *(Note: This directly overrides the standard "Batch count" in the core UI based on the number of checkpoints found in the folder.)*
6. Add your wildcards, select any other Main Script you want, and hit **Generate**.

## Note on Checkpoint Infotexts
This extension manually updates the `p.override_settings['sd_model_checkpoint']` metadata during each model swap, ensuring your final generated images correctly save the metadata for the *exact* model that generated them.
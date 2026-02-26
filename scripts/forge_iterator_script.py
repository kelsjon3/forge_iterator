import gradio as gr
import modules.scripts as scripts
import modules.shared as shared
from modules.processing import process_images
from modules.sd_models import checkpoints_list
import modules.sd_models

class ForgeIteratorScript(scripts.Script):
    def title(self):
        return "Forge Iterator"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def get_subfolders(self):
        # Scan checkpoints_list (which is a dict of {title: CheckpointInfo})
        # Extract the directory paths relative to the base model directory
        subfolders = set()
        for ckpt in modules.sd_models.checkpoints_list.values():
            if not ckpt.filename:
                continue
            
            # The CheckpointInfo object in sd_models has a filename. 
            # Often, ckpt.name is the relative path from the models/Stable-diffusion dir
            # e.g., "SubfolderA/my_model.safetensors"
            path_parts = ckpt.name.split('/') # or '\\' depending on OS, name usually uses '/' from a1111 logic
            if '\\' in ckpt.name:
                path_parts = ckpt.name.split('\\')
                
            if len(path_parts) > 1:
                # Reconstruct the subfolder path
                subfolder_path = '/'.join(path_parts[:-1])
                subfolders.add(subfolder_path)
                
        # Return a sorted list with an empty option for root/none selected
        choices = [""] + sorted(list(subfolders))
        return choices

    def ui(self, is_img2img):
        choices = self.get_subfolders()
        
        with gr.Accordion("Forge Iterator", open=False):
            enabled = gr.Checkbox(label="Enable Forge Iterator", value=False)
            
            # Use a refresh button to rescan folders if models are reloaded
            with gr.Row():
                folder = gr.Dropdown(label="Checkpoint Subfolder", choices=choices, value="")
                refresh_btn = gr.Button(value="🔄", size="sm", elem_classes="tool")
                
            quantity = gr.Slider(label="Iterations (Batches) per Checkpoint", minimum=1, maximum=100, step=1, value=1)

            def refresh_folders():
                # Modules.sd_models.list_models() updates the checkpoints_list under the hood 
                # if the user pressed the main UI refresh button before, but we just read the dict
                new_choices = self.get_subfolders()
                return gr.Dropdown.update(choices=new_choices)
                
            refresh_btn.click(fn=refresh_folders, outputs=[folder])

        return [enabled, folder, quantity]

    def _get_checkpoints_in_folder(self, folder):
        matches = []
        for ckpt_title, ckpt_info in modules.sd_models.checkpoints_list.items():
            name = ckpt_info.name
            # Normalize slashes for comparison
            normalized_name = name.replace('\\', '/')
            if normalized_name.startswith(f"{folder}/"):
                matches.append(ckpt_info)
                
        # Sort matches so order is deterministic (e.g., alphabetical)
        matches.sort(key=lambda x: x.name)
        return matches

    def setup(self, p, enabled, folder, quantity, **kwargs):
        if not enabled or not folder:
            return
            
        checkpoints_to_run = self._get_checkpoints_in_folder(folder)
        if not checkpoints_to_run:
            return
            
        # We perform n_iter inflation in setup() because it runs BEFORE Main Scripts (e.g. One Button Prompt).
        # This allows Prompt-Generating Main Scripts to correctly calculate how many dynamic prompts they need to make.
        p.n_iter = int(quantity) * len(checkpoints_to_run)
        
        # Initialize our absolute index tracker.
        # Main Scripts (like OBP) often set `p.n_iter = 1` and run their own `for` loops invoking `process_images` natively.
        # This makes kwargs.get('batch_number') statically 0 forever. We track it natively here.
        p.forge_iterator_current_index = 0

    def process(self, p, enabled, folder, quantity, **kwargs):
        if not enabled or not folder:
            return
            
        checkpoints_to_run = self._get_checkpoints_in_folder(folder)
        if not checkpoints_to_run:
            print(f"[Forge Iterator] No checkpoints found in folder: {folder}")
            return
            
        print(f"[Forge Iterator] Found {len(checkpoints_to_run)} checkpoints in {folder}. Multiplying batches.")
        
        # Save state for the batch loop
        p.forge_iterator_checkpoints = checkpoints_to_run
        p.forge_iterator_quantity = int(quantity)
        
        # Set the total number of iterations has been moved to setup() so Main Scripts see it early
        
        # Disable grid generation so we only get individual files per checkpoint iteration
        p.do_not_save_grid = True
        
        # We need to collect generated images across batches to return them to the UI
        p.forge_iterator_all_images = []
        p.forge_iterator_all_infotexts = []
        
        # Set the first model in the overrides so generation starts correctly natively
        first_ckpt = checkpoints_to_run[0]
        p.override_settings['sd_model_checkpoint'] = first_ckpt.title

    def process_batch(self, p, enabled, folder, quantity, **kwargs):
        if not enabled or not folder or not hasattr(p, 'forge_iterator_checkpoints'):
            return
            
        if not hasattr(p, 'forge_iterator_current_index'):
            p.forge_iterator_current_index = kwargs.get('batch_number', 0)
            
        current_index = p.forge_iterator_current_index
        p.forge_iterator_current_index += 1
        
        checkpoints_to_run = p.forge_iterator_checkpoints
        qty_per_ckpt = p.forge_iterator_quantity
        
        # Calculate which checkpoint we should be using
        # e.g., current_index=0, qty=2 -> index 0
        # current_index=1, qty=2 -> index 0
        # current_index=2, qty=2 -> index 1
        ckpt_index = current_index // qty_per_ckpt
        
        # Safety check
        if ckpt_index >= len(checkpoints_to_run):
            return
            
        target_ckpt = checkpoints_to_run[ckpt_index]
        
        # We need to swap the model if the currently loaded model isn't the target
        # Processed through override_settings on the FIRST batch
        # But mid-loop, we have to manually invoke the model reload
        current_ckpt_info = shared.sd_model.sd_checkpoint_info
        
        if current_ckpt_info.title != target_ckpt.title:
            print(f"[Forge Iterator] Swapping to checkpoint: {target_ckpt.name}")
            
            try:
                if hasattr(modules.sd_models, 'reload_model_weights'):
                    # A1111 Reload Logic
                    modules.sd_models.reload_model_weights(shared.sd_model, target_ckpt)
                else:
                    # Forge Reload Logic
                    modules.sd_models.model_data.forge_loading_parameters = dict(checkpoint_info=target_ckpt)
                    modules.sd_models.forge_model_reload()
                    
                p.sd_model = shared.sd_model
            except Exception as e:
                print(f"[Forge Iterator] Error swapping models: {e}")
                print(f"[Forge Iterator] Falling back to previous model: {current_ckpt_info.name}")
                
                # If Forge model reloading completely failed, shared.sd_model might be None
                # We need to attempt to reload the previous model to save the generation loop
                try:
                    if hasattr(modules.sd_models, 'reload_model_weights'):
                        modules.sd_models.reload_model_weights(shared.sd_model, current_ckpt_info)
                    else:
                        modules.sd_models.model_data.forge_loading_parameters = dict(checkpoint_info=current_ckpt_info)
                        modules.sd_models.forge_model_reload()
                    p.sd_model = shared.sd_model
                except Exception as fallback_e:
                    print(f"[Forge Iterator] Critical Error: Failed to restore fallback model: {fallback_e}")
                    pass
                
                # Instead of removing the corrupt model and breaking the array alignment, we simply replace the 
                # broken model in the checkpoints list with the healthy fallback model. This ensures remaining 
                # iterations of this batch and downstream scripts handle the transition seamlessly.
                print(f"[Forge Iterator] Marking corrupted model {target_ckpt.name} as failed. Replacing with fallback in rotation.")
                p.forge_iterator_checkpoints[ckpt_index] = current_ckpt_info
                
                # Important: If we fell back successfully, update the override_settings to the healthy model!
                # If we don't do this, downstream features like Hires Fix will accidentally try to load the corrupt model again.
                if hasattr(p, 'sd_model') and p.sd_model and hasattr(p.sd_model, 'sd_checkpoint_info'):
                    fallback_title = p.sd_model.sd_checkpoint_info.title
                    p.override_settings['sd_model_checkpoint'] = fallback_title
                    shared.opts.data['sd_model_checkpoint'] = fallback_title
                return
            
            # Ensure the overriding settings have the newly swapped title so Infotext saves correctly
            p.override_settings['sd_model_checkpoint'] = target_ckpt.title
            
            # Also update shared.opts data immediately because `set_config` relies on it mid-generation (Hires Fix, etc.)
            shared.opts.data['sd_model_checkpoint'] = target_ckpt.title

    def postprocess_image(self, p, pp, *args):
        # We collect each individual image as it finishes so we can return them all to the UI at the end
        if hasattr(p, 'forge_iterator_all_images'):
            p.forge_iterator_all_images.append(pp.image)
            
        # The user noted that the Live Preview frame doesn't show the final 100% VAE-decoded image
        # because the loop moves straight into the next batch. We force the live preview to display the
        # completed image by assigning it explicitly to the shared state.
        shared.state.assign_current_image(pp.image)

    def postprocess(self, p, processed, *args):
        # By default, when do_not_save_grid is True and n_iter is large, the WebUI 
        # sometimes fails to return the full list of generated images to the gallery preview.
        # We force the collected images into the processed object to ensure they display.
        if hasattr(p, 'forge_iterator_all_images') and p.forge_iterator_all_images:
            # We replace the processed images with our accumulated list
            if len(processed.images) < len(p.forge_iterator_all_images):
                processed.images = p.forge_iterator_all_images
                
                # Also need to extend infotexts to match the length of images so the UI doesn't crash reading metadata
                if len(processed.infotexts) < len(processed.images):
                    padding = [processed.infotexts[0] if processed.infotexts else ""] * (len(processed.images) - len(processed.infotexts))
                    processed.infotexts.extend(padding)
                
                # Ensure the UI gallery index starts at 0 since there is no Grid at index 0
                processed.index_of_first_image = 0

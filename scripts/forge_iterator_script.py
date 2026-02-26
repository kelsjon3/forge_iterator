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

    def process(self, p, enabled, folder, quantity):
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
        
        # Calculate total iterations
        # If user asked for n_iter=2 in the main UI, and we have 3 checkpoints and quantity=1
        # Should we respect n_iter?
        # Set the total number of iterations
        p.n_iter = p.forge_iterator_quantity * len(checkpoints_to_run)
        
        # Disable grid generation so we only get individual files per checkpoint iteration
        p.do_not_save_grid = True
        
        # Set the first model in the overrides so generation starts correctly natively
        first_ckpt = checkpoints_to_run[0]
        p.override_settings['sd_model_checkpoint'] = first_ckpt.title

    def process_batch(self, p, enabled, folder, quantity, **kwargs):
        if not enabled or not folder or not hasattr(p, 'forge_iterator_checkpoints'):
            return
            
        batch_number = kwargs.get('batch_number', 0)
        
        checkpoints_to_run = p.forge_iterator_checkpoints
        qty_per_ckpt = p.forge_iterator_quantity
        
        # Calculate which checkpoint we should be using
        # e.g., batch_number=0, qty=2 -> index 0
        # batch_number=1, qty=2 -> index 0
        # batch_number=2, qty=2 -> index 1
        ckpt_index = batch_number // qty_per_ckpt
        
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
                
                # To ensure we MOVE ON and don't duplicate generations, we must completely remove the broken target target_ckpt
                # from our list so the next batch calculation shifts to the *next* healthy model.
                print(f"[Forge Iterator] Removing corrupted model {target_ckpt.name} from rotation.")
                p.forge_iterator_checkpoints.remove(target_ckpt)
                
                # We must also decrease p.n_iter (total batches) natively so the loop doesn't spin empty cycles at the end
                p.n_iter -= qty_per_ckpt
                shared.state.job_count = p.n_iter 
                
                # Important: Do not update overriding settings with the target_ckpt if it failed
                return
            
            # Ensure the overriding settings have the newly swapped title so Infotext saves correctly
            p.override_settings['sd_model_checkpoint'] = target_ckpt.title

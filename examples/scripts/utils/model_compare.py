import torch

def compare_models(new_model, model, atol=1e-6, rtol=1e-5):
        sd_new = new_model.state_dict()
        sd_old = model.state_dict()

        # First check key sets match
        keys_new = set(sd_new.keys())
        keys_old = set(sd_old.keys())
        if keys_new != keys_old:
            missing_in_new = keys_old - keys_new
            missing_in_old = keys_new - keys_old
            print("Key mismatch detected!")
            if missing_in_new:
                print(f"  Missing in new_model: {sorted(missing_in_new)}")
            if missing_in_old:
                print(f"  Missing in model: {sorted(missing_in_old)}")
            return False

        # Now check values
        all_match = True
        for k in sd_new:
            v_new, v_old = sd_new[k].detach().cpu(), sd_old[k].detach().cpu()
            if not torch.allclose(v_new, v_old, atol=atol, rtol=rtol):
                all_match = False
                max_diff = (v_new - v_old).abs().max().item()
                print(f"Value mismatch at key: {k}, max diff: {max_diff}")
        if all_match:
            print("✅ All parameters match exactly within tolerance.")
        else:
            print("⚠️ Some parameters differ.")
        return all_match
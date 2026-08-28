import torch
import esm

model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
batch_converter = alphabet.get_batch_converter()
data = [("p1", "MKTLLILAVL"), ("p2", "MKTAYIAKQR")]
_, _, tokens = batch_converter(data)
with torch.no_grad():
    results = model(tokens, repr_layers=[6], return_contacts=False)
rep = results["representations"][6]
print("OK rep shape:", rep.shape)

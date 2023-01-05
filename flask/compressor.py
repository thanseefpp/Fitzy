# import pickle
# import bz2file as bz2

# def compressed_pickle(title, data):
#     with bz2.BZ2File(title + ".pbz2", "w") as f:
#         pickle.dump(data, f)
        
# huge_model = model
# compressed_pickle("compressed_model", model)
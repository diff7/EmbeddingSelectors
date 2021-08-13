import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms


def convert_and_normalize(file_name, size):
    if ".npy" in file_name:
        input_image = np.load(file_name)
        input_image = transforms.ToPILImage()(input_image)
    else:
        try:
            input_image = Image.open(file_name).convert("RGB")
        except:
            print(file_name)

    preprocess = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    tensor = preprocess(input_image).unsqueeze(0)
    return tensor


def get_embeddings_slow(
    model,
    data_txt_path,
    files=None,
    resize=32,
    model_name="",
    force_new=False,
):

    if files is None:
        with open(data_txt_path, "r") as f:
            files = f.read()
        files = [f.split(" ") for f in files.split("\n")]
    print(files[:10])

    path = (
        ["/"] + data_txt_path.split("/")[:-1] + [f"embeddings_{model_name}.npy"]
    )

    save_path = os.path.join(*path)
    print(f"Embeddings path {save_path}")
    if os.path.exists(save_path) and not force_new:
        print("Emb exists")
        embeddings = np.load(save_path)
        assert embeddings.shape[0] == len(
            files
        ), f"Different number of files found for the dataset num. embs: {embeddings.shape[0]} , train size {len(files)}"
        return embeddings

    embeddings = []
    model.cuda()
    model.eval()
    print("Making new embeddings")
    with torch.no_grad():
        for i, file_name in tqdm(enumerate(files)):
            if len(file_name) > 1:
                tensor = convert_and_normalize(file_name, resize)
                embedding = model(tensor.to("cuda"))
                embeddings.append(embedding.cpu())

    embeddings = np.array(torch.cat(embeddings))
    print(
        f"Saving embeddings to {save_path} embeddings shape: {embeddings.shape}"
    )
    np.save(save_path, embeddings)
    return embeddings


def get_resnet():
    from resnet import resnet20

    resnet = resnet20()
    backbone = torch.nn.Sequential(
        *list(resnet.children())[:-1],
        torch.nn.AdaptiveAvgPool2d(1),
    )
    backbone.load_state_dict(torch.load("resnet20-backbone-cifar10.state_dict"))
    return backbone


def get_VGG():
    import torchvision.models.vgg as models

    vgg16 = models.vgg16(pretrained=True)

    return torch.nn.Sequential(
        vgg16.features[:3], torch.nn.AdaptiveAvgPool2d(1)
    )


if __name__ == "__main__":
    FOLDER = "/home/dev/data_main/DIV2K/processed_50/LR/train/"
    files = sorted(
        [os.path.join(FOLDER, f) for f in os.listdir(FOLDER) if ".npy" not in f]
    )
    backbone = get_resnet()
    for f in files:
        if ".npy" in f:
            print(f)

    embeddings = get_embeddings_slow(
        backbone,
        data_txt_path=FOLDER,
        files=files,
        model_name="CIFAR_unsuper",
        resize=32,
        force_new=True,
    )

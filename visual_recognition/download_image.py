import requests

URL = "https://visualgenome.org/api/v0/images/"


def download_visual_genome_image(image_id, folder):
    """ Downloads image from the visual genome dataset

    :param image_id: id of the image to download
    :type image_id: str
    :param folder: where to download the image
    :return:
    """
    r = requests.get(url=URL + image_id)
    data = r.json()

    print(f"Downloading image: {data['url']}")

    img_data = requests.get(data['url']).content
    with open(folder + f'/{image_id}.jpg', 'wb') as handler:
        handler.write(img_data)


if __name__ == '__main__':
    download_visual_genome_image('1', '../data/visual_genome')

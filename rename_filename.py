import os

def get_unique_filename(folder_path, filename):
    """ Aynı isimde dosya varsa sonuna numara ekleyerek benzersiz hale getirir. """
    base, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename

    while os.path.exists(os.path.join(folder_path, new_filename)):  # Eğer dosya zaten varsa
        new_filename = f"{base}_{counter}{ext}"  # Örn: 1_1.txt, 1_2.txt
        counter += 1

    return new_filename

def rename_files_in_folder(folder_path):
    # Klasördeki tüm dosyaları al
    files = os.listdir(folder_path)

    # Desteklenen dosya uzantıları
    image_extensions = ('.png', '.jpg', '.jpeg')

    # Dosya tiplerine göre filtreleme ve sıralama
    image_files = sorted([f for f in files if f.lower().endswith(image_extensions)])
    txt_files = sorted([f for f in files if f.lower().endswith('.txt')])

    # Dosya sayılarının eşleştiğinden emin ol
    min_length = min(len(image_files), len(txt_files))

    # Yeniden adlandırma işlemi
    for index in range(min_length):
        image_old_name = image_files[index]
        txt_old_name = txt_files[index]

        # Yeni dosya isimlerini oluştur
        image_ext = os.path.splitext(image_old_name)[1]  # Orijinal uzantıyı koru (.png, .jpg, .jpeg)
        new_image_name = f"{index+1}{image_ext}"
        new_txt_name = f"{index+1}.txt"

        # Eğer aynı isimde dosya varsa benzersiz isim al
        new_image_name = get_unique_filename(folder_path, new_image_name)
        new_txt_name = get_unique_filename(folder_path, new_txt_name)

        # Dosyaları yeniden adlandır
        os.rename(os.path.join(folder_path, image_old_name), os.path.join(folder_path, new_image_name))
        os.rename(os.path.join(folder_path, txt_old_name), os.path.join(folder_path, new_txt_name))

        print(f"{image_old_name} -> {new_image_name}")
        print(f"{txt_old_name} -> {new_txt_name}")

# Kullanım
images_folder = r"D:\Atik Nakit\billboard\images"  # Klasör yolunu buraya gir
rename_files_in_folder(images_folder)

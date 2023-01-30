ImageNetS50="https://github.com/LUSSeg/ImageNet-S/releases/download/ImageNet-S/ImageNetS50-a0fe9d82231f9bc4787ee76e304dfa51.zip"
ImageNetS300="https://github.com/LUSSeg/ImageNet-S/releases/download/ImageNet-S/ImageNetS300-dd8e42e156bac415a35c3c239de297bf.zip"
ImageNetS919="https://github.com/LUSSeg/ImageNet-S/releases/download/ImageNet-S/ImageNetS919-5f7f58ae1003d21da9409a8576bf7680.zip"

cd $1
if [ "$2" = "50" ]
then
  wget ${ImageNetS50}
  unzip ImageNetS50-a0fe9d82231f9bc4787ee76e304dfa51.zip
fi

if [ "$2" = "300" ]
then
  wget ${ImageNetS300}
  unzip ImageNetS300-dd8e42e156bac415a35c3c239de297bf.zip
fi

if [ "$2" = "919" ]
then
  wget ${ImageNetS919}
  unzip ImageNetS919-5f7f58ae1003d21da9409a8576bf7680.zip
fi

if [ "$2" = "all" ]
then
  wget ${ImageNetS50}
  unzip ImageNetS50-a0fe9d82231f9bc4787ee76e304dfa51.zip
  wget ${ImageNetS300}
  unzip ImageNetS300-dd8e42e156bac415a35c3c239de297bf.zip
  wget ${ImageNetS919}
  unzip ImageNetS919-5f7f58ae1003d21da9409a8576bf7680.zip
fi

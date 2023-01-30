from torch.utils import data
from torchvision.datasets import ImageFolder
import torch
import os
from PIL import Image
import numpy as np
import argparse
from tqdm import tqdm
from munkres import Munkres
import multiprocessing
import json
from multiprocessing import Process, Manager
import data.utils as utils


class EvalDataset(ImageFolder):
    def __init__(self, pred_root, label_root, name_list, match=None):
        super(EvalDataset, self).__init__(label_root)

        self.pred_lst = []
        self.label_lst = []

        if match is not None:
            with open(match, 'r') as f:
                match = json.loads(f.read())
                match = {int(k): int(v) for k, v in match.items()}
        self.match = match

        with open(name_list, 'r') as f:
            names = f.read().splitlines()

            for name in names:
                name = name.split(' ')[1]
                self.label_lst.append(os.path.join(label_root, name))
                self.pred_lst.append(os.path.join(pred_root, name))

        self.classes_50 = "goldfish, tiger shark, goldfinch, tree frog, kuvasz, red fox, siamese cat, american black bear, ladybug, sulphur butterfly, wood rabbit, hamster, wild boar, gibbon, african elephant, giant panda, airliner, ashcan, ballpoint, beach wagon, boathouse, bullet train, cellular telephone, chest, clog, container ship, digital watch, dining table, golf ball, grand piano, iron, lab coat, mixing bowl, motor scooter, padlock, park bench, purse, streetcar, table lamp, television, toilet seat, umbrella, vase, water bottle, water tower, yawl, street sign, lemon, carbonara, agaric"
        self.classes_300 = "tench, goldfish, tiger shark, hammerhead, electric ray, ostrich, goldfinch, house finch, indigo bunting, kite, common newt, axolotl, tree frog, tailed frog, mud turtle, banded gecko, american chameleon, whiptail, african chameleon, komodo dragon, american alligator, triceratops, thunder snake, ringneck snake, king snake, rock python, horned viper, harvestman, scorpion, garden spider, tick, african grey, lorikeet, red-breasted merganser, wallaby, koala, jellyfish, sea anemone, conch, fiddler crab, american lobster, spiny lobster, isopod, bittern, crane, limpkin, bustard, albatross, toy terrier, afghan hound, bluetick, borzoi, irish wolfhound, whippet, ibizan hound, staffordshire bullterrier, border terrier, yorkshire terrier, lakeland terrier, giant schnauzer, standard schnauzer, scotch terrier, lhasa, english setter, clumber, english springer, welsh springer spaniel, kuvasz, kelpie, doberman, miniature pinscher, malamute, pug, leonberg, great pyrenees, samoyed, brabancon griffon, cardigan, coyote, red fox, kit fox, grey fox, persian cat, siamese cat, cougar, lynx, tiger, american black bear, sloth bear, ladybug, leaf beetle, weevil, bee, cicada, leafhopper, damselfly, ringlet, cabbage butterfly, sulphur butterfly, sea cucumber, wood rabbit, hare, hamster, wild boar, hippopotamus, bighorn, ibex, badger, three-toed sloth, orangutan, gibbon, colobus, spider monkey, squirrel monkey, madagascar cat, indian elephant, african elephant, giant panda, barracouta, eel, coho, academic gown, accordion, airliner, ambulance, analog clock, ashcan, backpack, balloon, ballpoint, barbell, barn, bassoon, bath towel, beach wagon, bicycle-built-for-two, binoculars, boathouse, bonnet, bookcase, bow, brass, breastplate, bullet train, cannon, can opener, carpenter's kit, cassette, cellular telephone, chain saw, chest, china cabinet, clog, combination lock, container ship, corkscrew, crate, crock pot, digital watch, dining table, dishwasher, doormat, dutch oven, electric fan, electric locomotive, envelope, file, folding chair, football helmet, freight car, french horn, fur coat, garbage truck, goblet, golf ball, grand piano, half track, hamper, hard disc, harmonica, harvester, hook, horizontal bar, horse cart, iron, jack-o'-lantern, lab coat, ladle, letter opener, liner, mailbox, megalith, military uniform, milk can, mixing bowl, monastery, mortar, mosquito net, motor scooter, mountain bike, mountain tent, mousetrap, necklace, nipple, ocarina, padlock, palace, parallel bars, park bench, pedestal, pencil sharpener, pickelhaube, pillow, planetarium, plastic bag, polaroid camera, pole, pot, purse, quilt, radiator, radio, radio telescope, rain barrel, reflex camera, refrigerator, rifle, rocking chair, rubber eraser, rule, running shoe, sewing machine, shield, shoji, ski, ski mask, slot, soap dispenser, soccer ball, sock, soup bowl, space heater, spider web, spindle, sports car, steel arch bridge, stethoscope, streetcar, submarine, swimming trunks, syringe, table lamp, tank, teddy, television, throne, tile roof, toilet seat, trench coat, trimaran, typewriter keyboard, umbrella, vase, volleyball, wardrobe, warplane, washer, water bottle, water tower, whiskey jug, wig, wine bottle, wok, wreck, yawl, yurt, street sign, traffic light, consomme, ice cream, bagel, cheeseburger, hotdog, mashed potato, spaghetti squash, bell pepper, cardoon, granny smith, strawberry, lemon, carbonara, burrito, cup, coral reef, yellow lady's slipper, buckeye, agaric, gyromitra, earthstar, bolete"
        self.classes_919 = "house finch, stupa, agaric, hen-of-the-woods, wild boar, kit fox, desk, beaker, spindle, lipstick, cardoon, ringneck snake, daisy, sturgeon, scorpion, pelican, bustard, rock crab, rock beauty, minivan, menu, thunder snake, zebra, partridge, lacewing, starfish, italian greyhound, marmot, cardigan, plate, ballpoint, chesapeake bay retriever, pirate, potpie, keeshond, dhole, waffle iron, cab, american egret, colobus, radio telescope, gordon setter, mousetrap, overskirt, hamster, wine bottle, bluetick, macaque, bullfrog, junco, tusker, scuba diver, pool table, samoyed, mailbox, purse, monastery, bathtub, window screen, african crocodile, traffic light, tow truck, radio, recreational vehicle, grey whale, crayfish, rottweiler, racer, whistle, pencil box, barometer, cabbage butterfly, sloth bear, rhinoceros beetle, guillotine, rocking chair, sports car, bouvier des flandres, border collie, fiddler crab, slot, go-kart, cocker spaniel, plate rack, common newt, tile roof, marimba, moped, terrapin, oxcart, lionfish, bassinet, rain barrel, american black bear, goose, half track, kite, microphone, shield, mexican hairless, measuring cup, bubble, platypus, saint bernard, police van, vase, lhasa, wardrobe, teapot, hummingbird, revolver, jinrikisha, mailbag, red-breasted merganser, assault rifle, loudspeaker, fig, american lobster, can opener, arctic fox, broccoli, long-horned beetle, television, airship, black stork, marmoset, panpipe, drumstick, knee pad, lotion, french loaf, throne, jeep, jersey, tiger cat, cliff, sealyham terrier, strawberry, minibus, goldfinch, goblet, burrito, harp, tractor, cornet, leopard, fly, fireboat, bolete, barber chair, consomme, tripod, breastplate, pineapple, wok, totem pole, alligator lizard, common iguana, digital clock, bighorn, siamese cat, bobsled, irish setter, zucchini, crock pot, loggerhead, irish wolfhound, nipple, rubber eraser, impala, barbell, snow leopard, siberian husky, necklace, manhole cover, electric fan, hippopotamus, entlebucher, prison, doberman, ruffed grouse, coyote, toaster, puffer, black swan, schipperke, file, prairie chicken, hourglass, greater swiss mountain dog, pajama, ear, pedestal, viaduct, shoji, snowplow, puck, gyromitra, birdhouse, flatworm, pier, coral reef, pot, mortar, polaroid camera, passenger car, barracouta, banded gecko, black-and-tan coonhound, safe, ski, torch, green lizard, volleyball, brambling, solar dish, lawn mower, swing, hyena, staffordshire bullterrier, screw, toilet tissue, velvet, scale, stopwatch, sock, koala, garbage truck, spider monkey, afghan hound, chain, upright, flagpole, tree frog, cuirass, chest, groenendael, christmas stocking, lakeland terrier, perfume, neck brace, lab coat, carbonara, porcupine, shower curtain, slug, pitcher, flat-coated retriever, pekinese, oscilloscope, church, lynx, cowboy hat, table lamp, pug, crate, water buffalo, labrador retriever, weimaraner, giant schnauzer, stove, sea urchin, banjo, tiger, miniskirt, eft, european gallinule, vending machine, miniature schnauzer, maypole, bull mastiff, hoopskirt, coffeepot, four-poster, safety pin, monarch, beer glass, grasshopper, head cabbage, parking meter, bonnet, chiffonier, great dane, spider web, electric locomotive, scotch terrier, australian terrier, honeycomb, leafhopper, beer bottle, mud turtle, lifeboat, cassette, potter's wheel, oystercatcher, space heater, coral fungus, sunglass, quail, triumphal arch, collie, walker hound, bucket, bee, komodo dragon, dugong, gibbon, trailer truck, king crab, cheetah, rifle, stingray, bison, ipod, modem, box turtle, motor scooter, container ship, vestment, dingo, radiator, giant panda, nail, sea slug, indigo bunting, trimaran, jacamar, chimpanzee, comic book, odometer, dishwasher, bolo tie, barn, paddlewheel, appenzeller, great white shark, green snake, jackfruit, llama, whippet, hay, leaf beetle, sombrero, ram, washbasin, cup, wall clock, acorn squash, spotted salamander, boston bull, border terrier, doormat, cicada, kimono, hand blower, ox, meerkat, space shuttle, african hunting dog, violin, artichoke, toucan, bulbul, coucal, red wolf, seat belt, bicycle-built-for-two, bow tie, pretzel, bedlington terrier, albatross, punching bag, cocktail shaker, diamondback, corn, ant, mountain bike, walking stick, standard schnauzer, power drill, cardigan, accordion, wire-haired fox terrier, streetcar, beach wagon, ibizan hound, hair spray, car mirror, mountain tent, trench coat, studio couch, pomeranian, dough, corkscrew, broom, parachute, band aid, water tower, teddy, fire engine, hornbill, hotdog, theater curtain, crane, malinois, lion, african elephant, handkerchief, caldron, shopping basket, gown, wolf spider, vizsla, electric ray, freight car, pembroke, feather boa, wallet, agama, hard disc, stretcher, sorrel, trilobite, basset, vulture, tarantula, hermit crab, king snake, robin, bernese mountain dog, ski mask, fountain pen, combination lock, yurt, clumber, park bench, baboon, kuvasz, centipede, tabby, steam locomotive, badger, irish water spaniel, picket fence, gong, canoe, swimming trunks, submarine, echidna, bib, refrigerator, hammer, lemon, admiral, chihuahua, basenji, pinwheel, golfcart, bullet train, crib, muzzle, eggnog, old english sheepdog, tray, tiger beetle, electric guitar, peacock, soup bowl, wallaby, abacus, dalmatian, harvester, aircraft carrier, snowmobile, welsh springer spaniel, affenpinscher, oboe, cassette player, pencil sharpener, japanese spaniel, plunger, black widow, norfolk terrier, reflex camera, ice bear, redbone, mongoose, warthog, arabian camel, bittern, mixing bowl, tailed frog, scabbard, castle, curly-coated retriever, garden spider, folding chair, mouse, prayer rug, red fox, toy terrier, leonberg, lycaenid, poncho, goldfish, red-backed sandpiper, holster, hair slide, coho, komondor, macaw, maltese dog, megalith, sarong, green mamba, sea lion, water ouzel, bulletproof vest, sulphur-crested cockatoo, scottish deerhound, steel arch bridge, catamaran, brittany spaniel, redshank, otter, brabancon griffon, balloon, rule, planetarium, trombone, mitten, abaya, crash helmet, milk can, hartebeest, windsor tie, irish terrier, african chameleon, matchstick, water bottle, cloak, ground beetle, ashcan, crane, gila monster, unicycle, gazelle, wombat, brain coral, projector, custard apple, proboscis monkey, tibetan mastiff, mosque, plastic bag, backpack, drum, norwich terrier, pizza, carton, plane, gorilla, jigsaw puzzle, forklift, isopod, otterhound, vacuum, european fire salamander, apron, langur, boxer, african grey, ice lolly, toilet seat, golf ball, titi, drake, ostrich, magnetic compass, great pyrenees, rhodesian ridgeback, buckeye, dungeness crab, toy poodle, ptarmigan, amphibian, monitor, school bus, schooner, spatula, weevil, speedboat, sundial, borzoi, bassoon, bath towel, pill bottle, acorn, tick, briard, thimble, brass, white wolf, boathouse, yawl, miniature pinscher, barn spider, jean, water snake, dishrag, yorkshire terrier, hammerhead, typewriter keyboard, papillon, ocarina, washer, standard poodle, china cabinet, steel drum, swab, mobile home, german short-haired pointer, saluki, bee eater, rock python, vine snake, kelpie, harmonica, military uniform, reel, thatch, maraca, tricycle, sidewinder, parallel bars, banana, flute, paintbrush, sleeping bag, yellow lady's slipper, three-toed sloth, white stork, notebook, weasel, tiger shark, football helmet, madagascar cat, dowitcher, wreck, king penguin, lighter, timber wolf, racket, digital watch, liner, hen, suspension bridge, pillow, carpenter's kit, butternut squash, sandal, sussex spaniel, hip, american staffordshire terrier, flamingo, analog clock, black and gold garden spider, sea cucumber, indian elephant, syringe, lens cap, missile, cougar, diaper, chambered nautilus, garter snake, anemone fish, organ, limousine, horse cart, jaguar, frilled lizard, crutch, sea anemone, guenon, meat loaf, slide rule, saltshaker, pomegranate, acoustic guitar, shopping cart, drilling platform, nematode, chickadee, academic gown, candle, norwegian elkhound, armadillo, horizontal bar, orangutan, obelisk, stone wall, cannon, rugby ball, ping-pong ball, window shade, trolleybus, ice cream, pop bottle, cock, harvestman, leatherback turtle, killer whale, spaghetti squash, chain saw, stinkhorn, espresso maker, loafer, bagel, ballplayer, skunk, chainlink fence, earthstar, whiptail, barrel, kerry blue terrier, triceratops, chow, grey fox, sax, binoculars, ladybug, silky terrier, gas pump, cradle, whiskey jug, french bulldog, eskimo dog, hog, hognose snake, pickup, indian cobra, hand-held computer, printer, pole, bald eagle, american alligator, dumbbell, umbrella, mink, shower cap, tank, quill, fox squirrel, ambulance, lesser panda, frying pan, letter opener, hook, strainer, pick, dragonfly, gar, piggy bank, envelope, stole, ibex, american chameleon, bearskin, microwave, petri dish, wood rabbit, beacon, dung beetle, warplane, ruddy turnstone, knot, fur coat, hamper, beagle, ringlet, mask, persian cat, cellular telephone, american coot, apiary, shovel, coffee mug, sewing machine, spoonbill, padlock, bell pepper, great grey owl, squirrel monkey, sulphur butterfly, scoreboard, bow, malamute, siamang, snail, remote control, sea snake, loupe, model t, english setter, dining table, face powder, tench, jack-o'-lantern, croquet ball, water jug, airedale, airliner, guinea pig, hare, damselfly, thresher, limpkin, buckle, english springer, boa constrictor, french horn, black-footed ferret, shetland sheepdog, capuchin, cheeseburger, miniature poodle, spotlight, wooden spoon, west highland white terrier, wig, running shoe, cowboy boot, brown bear, iron, brassiere, magpie, gondola, grand piano, granny smith, mashed potato, german shepherd, stethoscope, cauliflower, soccer ball, pay-phone, jellyfish, cairn, polecat, trifle, photocopier, shih-tzu, orange, guacamole, hatchet, cello, egyptian cat, basketball, moving van, mortarboard, dial telephone, street sign, oil filter, beaver, spiny lobster, chime, bookcase, chiton, black grouse, jay, axolotl, oxygen mask, cricket, worm fence, indri, cockroach, mushroom, dandie dinmont, tennis ball, howler monkey, rapeseed, tibetan terrier, newfoundland, dutch oven, paddle, joystick, golden retriever, blenheim spaniel, mantis, soft-coated wheaten terrier, little blue heron, convertible, bloodhound, palace, medicine chest, english foxhound, cleaver, sweatshirt, mosquito net, soap dispenser, ladle, screwdriver, fire screen, binder, suit, barrow, clog, cucumber, baseball, lorikeet, conch, quilt, eel, horned viper, night snake, angora, pickelhaube, gasmask, patas"
        self.classes_50 = ['background'] + self.classes_50.split(', ')
        self.classes_300 = ['background'] + self.classes_300.split(', ')
        self.classes_919 = ['background'] + self.classes_919.split(', ')

    def __getitem__(self, item):
        gt = Image.open(self.label_lst[item])
        gt = np.array(gt)
        gt = gt[:, :, 1] * 256 + gt[:, :, 0]

        predict = np.array(Image.open(os.path.join(self.pred_lst[item])))
        predict = predict[:, :, 1] * 256 + predict[:, :, 0]

        # Get boundary mask for each class.
        boundary_gt = self.get_boundary_mask(gt + 1)
        boundary_predict = self.get_boundary_mask(predict + 1)

        gt = torch.from_numpy(gt.astype(np.float))
        predict = torch.from_numpy(predict.astype(np.float))
        boundary_gt = torch.from_numpy(boundary_gt.astype(np.float))
        boundary_predict = torch.from_numpy(boundary_predict.astype(np.float))

        gt = gt.view(-1)
        predict = predict.view(-1)
        boundary_gt = boundary_gt.view(-1)
        boundary_predict = boundary_predict.view(-1)

        mask = gt != 1000
        gt = gt[mask]
        predict = predict[mask]
        boundary_gt = boundary_gt[mask]
        boundary_predict = boundary_predict[mask]

        if self.match is not None:
            predict_match = torch.zeros_like(predict)
            boundary_predict_match = torch.zeros_like(boundary_predict)
            for v in torch.unique(predict):
                predict_match[predict == v] = self.match[v.item()]
                boundary_predict_match[boundary_predict == v + 1] = self.match[v.item()] + 1

            return gt, predict_match, boundary_gt, boundary_predict_match

        return gt, predict, boundary_gt, boundary_predict
    
    def get_boundary_mask(self, mask):
        boundary = np.zeros_like(mask).astype(mask.dtype)
        for v in np.unique(mask):
            mask_v = utils.get_mask_of_class(mask, v)
            boundary_v = utils.mask_to_boundary(mask_v, dilation_ratio=0.03)
            boundary += (boundary_v > 0) * v
        return boundary

    def __len__(self):
        return len(self.label_lst)

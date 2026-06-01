from util import str2vec, cos_sim, dist
from icecream import ic

if __name__ == "__main__":
    hw1 = "Hello, world!"
    hw2 = "hello world"
    tiap1 = "This is a pen."
    tiap2 = "this is a pen"

    hw1_vec = str2vec(hw1)
    hw2_vec = str2vec(hw2)
    tiap1_vec = str2vec(tiap1)
    tiap2_vec = str2vec(tiap2)

    ic(dist(hw1_vec, hw2_vec))
    ic(dist(hw1_vec, tiap1_vec))
    ic(dist(hw1_vec, tiap2_vec))

    ic(cos_sim(hw1_vec, hw2_vec))
    ic(cos_sim(hw1_vec, tiap1_vec))
    ic(cos_sim(hw1_vec, tiap2_vec))

from util.vec import str2vec, cos_sim, dist
from icecream import ic

if __name__ == "__main__":
    # hm = "hello morning!"
    # gm = "goodbye morning!"
    # hn = "hello night!"
    # gn = "goodbye night!"

    hm = "king"
    gm = "queen"
    hn = "man"
    gn = "woman"

    hm_vec = str2vec(hm)
    gm_vec = str2vec(gm)
    hn_vec = str2vec(hn)
    gn_vec = str2vec(gn)

    ic(dist(hm_vec, gm_vec))
    ic(dist(hm_vec - hn_vec + gn_vec, gm_vec))

    ic(cos_sim(hm_vec, gm_vec))
    ic(cos_sim(hm_vec - hn_vec + gn_vec, gm_vec))

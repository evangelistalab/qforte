from qforte import manifolds


class TestManifolds:
    def test_manifolds(self):
        ref = [1] * 6 + [0] * 6
        irreps = [0, 1, 2, 2, 1, 0]

        cis_manifold = manifolds.cis_manifold(ref)
        assert len(cis_manifold) == 18

        cis_0_manifold = manifolds.cis_manifold(ref, irreps=irreps, target_irrep=0)
        assert len(cis_0_manifold) == 6

        cis_1_manifold = manifolds.cis_manifold(ref, irreps=irreps, target_irrep=1)
        assert len(cis_1_manifold) == 4

        cis_2_manifold = manifolds.cis_manifold(ref, irreps=irreps, target_irrep=2)
        assert len(cis_2_manifold) == 4

        cis_3_manifold = manifolds.cis_manifold(ref, irreps=irreps, target_irrep=3)
        assert len(cis_3_manifold) == 4

        cisd_manifold = manifolds.cisd_manifold(ref)
        assert len(cisd_manifold) == 117

        cisd_sym_manifold = []
        for i in range(4):
            cisd_sym_manifold += manifolds.cisd_manifold(
                ref, irreps=irreps, target_irrep=i
            )
        assert len(cisd_sym_manifold) == 117

        cisd_spin_flip_manifold = manifolds.cisd_manifold(ref, sz=[-2, -1, 0, 1, 2])
        assert len(cisd_spin_flip_manifold) == 261

        h2p_manifold = manifolds.ee_ip_ea_manifold(ref, 2, 1, sz=[-1.5, -0.5, 0.5, 1.5])
        assert len(h2p_manifold) == 90

        h2p_sym_manifold = []
        for i in range(4):
            h2p_sym_manifold += manifolds.ee_ip_ea_manifold(
                ref, 2, 1, sz=[-1.5, -0.5, 0.5, 1.5], irreps=irreps, target_irrep=i
            )
        assert len(h2p_sym_manifold) == 90

        p2h_manifold = manifolds.ee_ip_ea_manifold(ref, 1, 2, sz=[-1.5, -0.5, 0.5, 1.5])
        assert len(p2h_manifold) == 90

        hp2_sym_manifold = []
        for i in range(4):
            hp2_sym_manifold += manifolds.ee_ip_ea_manifold(
                ref, 1, 2, sz=[-1.5, -0.5, 0.5, 1.5], irreps=irreps, target_irrep=i
            )
        assert len(hp2_sym_manifold) == 90

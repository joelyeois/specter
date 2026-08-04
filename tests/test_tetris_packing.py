import torch

from specter.specimen.packing import (
    TetrisPackingSpecimenGenerator,
    TetrisProteinSpec,
)


def test_tetris_packing_places_template_from_contact_score():
    template = torch.zeros(7, 7, 7)
    template[2:5, 2:5, 2:5] = 200.0

    gen = TetrisPackingSpecimenGenerator(
        [TetrisProteinSpec("cube", frequency=1, template=template)],
        target_shape=(24, 24, 24),
        iterations=1,
        insertion_distances=(-1, 1),
        sigma=0,
        gray_level_threshold=50,
        progressbars=False,
        seed=0,
    )

    volume = gen.generate()

    assert volume.shape == (24, 24, 24)
    assert len(gen.placements) == 2
    assert gen.placements[0].center_zyx.tolist() == [12, 12, 12]
    assert gen.placements[1].score > 0
    assert volume.sum() > template.sum()


def test_tetris_packing_fft_backend_places_template():
    template = torch.zeros(7, 7, 7)
    template[2:5, 2:5, 2:5] = 200.0

    gen = TetrisPackingSpecimenGenerator(
        [TetrisProteinSpec("cube", frequency=1, template=template)],
        target_shape=(24, 24, 24),
        iterations=1,
        insertion_distances=(-1, 1),
        sigma=0,
        gray_level_threshold=50,
        correlation_backend="fft",
        progressbars=False,
        seed=0,
    )

    gen.generate()

    assert len(gen.placements) == 2
    assert gen.placements[1].score > 0


def test_tetris_packing_auto_threshold_handles_low_amplitude_templates():
    template = torch.zeros(7, 7, 7)
    template[2:5, 2:5, 2:5] = 5.0

    gen = TetrisPackingSpecimenGenerator(
        [TetrisProteinSpec("low_amp", frequency=1, template=template)],
        target_shape=(24, 24, 24),
        iterations=1,
        insertion_distances=(-1, 1),
        sigma=0,
        gray_level_threshold=100,
        threshold_mode="auto",
        progressbars=False,
        seed=0,
    )

    gen.generate()

    assert len(gen.placements) == 2
    assert "low_amp" not in gen.saturated_species


def test_tetris_packing_tiny_template_keeps_overlap_penalty_layer():
    template = torch.zeros(5, 5, 5)
    template[2, 2, 2] = 5.0

    gen = TetrisPackingSpecimenGenerator(
        [TetrisProteinSpec("tiny", frequency=1, template=template)],
        target_shape=(17, 17, 17),
        iterations=1,
        insertion_distances=(-2, 0),
        sigma=0,
        gray_level_threshold=100,
        threshold_mode="auto",
        progressbars=False,
        seed=0,
    )
    score_template = gen._make_score_template(template)

    assert (score_template < 0).any()


def test_tetris_packing_exports_oriented_picks(tmp_path):
    template = torch.zeros(7, 7, 7)
    template[2:5, 2:5, 2:5] = 200.0
    gen = TetrisPackingSpecimenGenerator(
        [TetrisProteinSpec("cube", template=template)],
        target_shape=(24, 24, 24),
        iterations=0,
        sigma=0,
        gray_level_threshold=50,
        v_size=2.0,
        progressbars=False,
        seed=0,
    )
    gen.generate()

    written = gen.export_picks(tmp_path)

    path = written["cube"]
    text = path.read_text()
    assert '"type": "orientedPoint"' in text
    assert '"x": 24.0' in text
    assert '"y": 24.0' in text
    assert '"z": 24.0' in text
    assert "xyz_rotation_matrix" in text

import pytest

from aurora_lens.interpret.spacy_backend import SpacyBackend
from aurora_lens.verify.checker import Checker
from aurora_lens.verify.flags import FlagType
from aurora_lens.verify.medical_directive_scope import treatment_bindings


@pytest.fixture(scope="module")
def backend():
    return SpacyBackend()


@pytest.mark.parametrize("text,medical", [
    ("You should ensure you understand the tax treatment. You can take loans from your fund.", False),
    ("You should take your medication. Understand the tax treatment of your fund.", True),
    ("You should take your medication and ensure you understand the tax treatment.", True),
    ("You should review your contract. You should stop your medication.", True),
    ("You should avoid sodium hydroxide in the laboratory. You should start your medication.", True),
    ("You should avoid sodium hydroxide in the laboratory.", False),
    ("Your tax treatment matters. You should take time to review fees.", False),
    ("Tell your doctor about your allergy. You should take your medication.", True),
    ("You should check with your doctor before taking your medication.", False),
    ("You should check with your doctor before taking medication, and you should stop your insulin.", True),
    ("This medication may help. You should take it.", True),
    ("You should take magnesium hydroxide for your symptoms.", True),
    ("You should take sodium chloride for your symptoms.", True),
])
def test_bound_medical_objects(backend, text, medical):
    flags = Checker(backend)._check_professional_directive(text)
    assert any(f.flag_type == FlagType.PERSONALIZED_MEDICAL_ADVICE for f in flags) == medical


def test_nonclinical_binding_is_observable(backend):
    bindings = treatment_bindings("You should review your tax treatment.", backend.nlp)
    assert bindings and not bindings[0].clinical
    assert bindings[0].basis == "explicit_nonclinical_object_modifier"


def test_unavailable_parser_is_unresolved():
    assert treatment_bindings("You should take medication.", None) is None

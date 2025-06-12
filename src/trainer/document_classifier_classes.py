from enum import Enum

class DocumentClassifierClasses(Enum):
    """
    Enum for document classification classes.
    """
    ATTESTATION_HEBERGEMENT = "attestation hebergement"
    AVIS_IMPOT_TAXE_FONCIERE = "avis impot taxe fonciere"
    AVIS_IMPOT_SUR_REVENUS = "avis sur revenus"
    BULLETIN_DE_SALAIRE = "bulletin de salaire"
    RELEVE_DE_COMPTE_BANCAIRE = "releve de compte bancaire"
    
    
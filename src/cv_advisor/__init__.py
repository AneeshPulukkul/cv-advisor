"""cv-advisor: recommend the right scikit-learn CV splitter for any tabular dataset."""

from cv_advisor.advisor import CVAdvisor
from cv_advisor.schema import CVRecommendation, DataProfile

__all__ = ["CVAdvisor", "CVRecommendation", "DataProfile"]
__version__ = "0.2.0"

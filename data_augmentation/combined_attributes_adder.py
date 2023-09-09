from sklearn.base import BaseEstimator, TransformerMixin


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, attr_to_add):
        self.attr_to_add = attr_to_add

    def fit(self, X, y=None):
        return self # no fitting required

    def transform(self, X, y=None):
        if self.attr_to_add['RoomsPerBedrm']:
            rooms_per_bedrm = X['AveRooms'] / X['AveBedrms']
            X['RoomsPerBedrm'] = rooms_per_bedrm
        if self.attr_to_add['MedIncPerPop']:
            med_inc_per_pop = X['MedInc'] / X['Population']
            X['MedIncPerPop'] = med_inc_per_pop
        if self.attr_to_add['AveOccupPerRoom']:
            ave_occup_per_room = X['AveOccup'] / X['AveRooms']
            X['AveOccupPerRoom'] = ave_occup_per_room
        if self.attr_to_add['MedIncPerRoom']:
            med_inc_per_room = X['MedInc'] / X['AveRooms']
            X['MedIncPerRoom'] = med_inc_per_room
        if self.attr_to_add['MedIncPerBedrm']:
            med_inc_per_bedrm = X['MedInc'] / X['AveBedrms']
            X['MedIncPerBedrm'] = med_inc_per_bedrm
        return X

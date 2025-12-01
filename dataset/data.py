class MIMIC4Data:

    def __init__(
            self,
            admission_id,
            patient_id,
            duration,
            mortality,
            readmission,
            age,
            gender,
            ethnicity,
    ):
        self.admission_id = admission_id  # str
        self.patient_id = patient_id  # str
        self.duration = duration  # int
        self.mortality = mortality  # bool, end of icu stay mortality
        self.readmission = readmission  # bool, 15-day icu readmission
        self.age = age  # int
        self.gender = gender  # str
        self.ethnicity = ethnicity  # str

        # list of tuples (timestamp in min (int), type (str), list of codes (str))
        self.diagnoses_icd = []
        self.procedures_icd = []
        self.prescriptions = []

        # (list of types (str), list of codes (str))
        self.trajectory = []

        # labs
        # (timestamp in min (int), list of (item_id, value, flag))
        self.labevents = []
        # numpy array
        self.labvectors = None

        # notes
        self.discharge = None

    def __repr__(self):
        return f"HOSP_ADM ID-{self.admission_id} ({self.duration} min): " \
               f"mortality-{self.mortality}, " \
               f"readmission-{self.readmission}"


class ADNIData:

    def __init__(
            self,
            id,
            age,
            gender,
            ethnicity,
    ):
        self.vid = id
        self.patient_id = id.split('-')[0]  # str
        self.age = age  # int
        self.gender = gender  # str
        self.ethnicity = ethnicity  # str
        self.x1 = None
        self.x2 = None
        self.x3 = None
        self.y = None

    def __repr__(self):
        return f"VID-{self.vid}: " \
               f"label-{self.y}"

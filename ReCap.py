from RelevanceEvluator import RelevanceEvaluator

class ReCap:
    def __init__(self, image, caption, clip_checkpoint="ViT-B/32", generic=False):
        self.ClipModel = RelevanceEvaluator(clip_checkpoint)
        self.object_reference = ['human', 'animal', 'machine', 'insect', 'tree', 'building', 'plant', 'food', 'tool', 'house']
        self.feature_reference = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'white', 'black', 'purple']
        self.object_reference_clip_score = dict()
        self.feature_reference_clip_score = dict()
        self.T_hat = None
        self.image = image
        self.caption = caption
        self.candidate_segments = None
        self.candidate_tokens = None
        self.candidate_clip_score = dict()
        self.target_tokens = dict()
        self.generic = generic

    def run(self):
        self.filteration()
        if not (self.candidate_segments and self.candidate_tokens):# and self.T_hat):
            print("Assign candidate lists!")
            return
        self.relevance_evaluation()
        self.substitution()

    def filteration(self):
        self.text_filteration()
        self.image_filteration()

    def text_filteration(self):
        # it returns a list of triplets
        # T_c = {'token_1':{t_1, l_1, g_1},.... }
        print("Assign candidate_tokens in form of T_c = {'token_0':{t_0, l_0, g_0},.... }")

    def image_filteration(self):
        # it returns a list of duals
        # I_c = {'segment_1':{i_1, l_1},.... }
        print("Assign candidate_segments in form of I_c = {'segment_0':{i_0, l_0},.... }")

    def relevance_evaluation(self):
        self.calculate_candidate_clip_score()
        self.calculate_reference_clip_score()
        self.generate_target_tokens()

    def calculate_reference_clip_score(self):
        for i_id in range(len(self.candidate_segments)):
            i_k = self.candidate_segments[f"segment_{i_id}"]
            segment, i_label = i_k
            for obj_id in range(len(self.object_reference)):
                obj_ref = self.object_reference[obj_id]
                clip_score = self.ClipModel.measure_similarity(segment, obj_ref)
                self.object_reference_clip_score[f"segment_{i_id}_obj_{obj_id}"] = clip_score
            for f_id in range(len(self.feature_reference)):
                f_ref = self.feature_reference[f_id]
                clip_score = self.ClipModel.measure_similarity(segment, f_ref)
                self.feature_reference_clip_score[f"segment_{i_id}_f_{f_id}"] = clip_score

    def calculate_candidate_clip_score(self):
        for i_id in range(len(self.candidate_segments)):
            i_k = self.candidate_segments[f"segment_{i_id}"]
            segment, i_label = i_k
            for t_id in range(len(self.candidate_tokens)):
                t_k = self.candidate_tokens[f"token_{t_id}"]
                token, t_label, g_label = t_k

                if i_label == t_label:
                    clip_score = self.ClipModel.measure_similarity(segment, token)
                    self.candidate_clip_score[f"segment_{i_id}_token_{t_id}"] = clip_score



    def generate_target_tokens(self):
        for c_id, c_score in self.candidate_clip_score.items():
            c_id_split = c_id.split("_")    # "segment_{i_k}_token_{t_k}" -> segment, i_k, token, t_k
            i_id, t_id = c_id_split[1],  c_id_split[3]
            g_k = self.candidate_tokens[f"token_{t_id}"][-1] # either f or obj
            if g_k == "f":
                reference_clip_score = self.feature_reference_clip_score.copy()
            elif g_k == "obj":
                reference_clip_score = self.object_reference_clip_score.copy()
            ref_thr, ref_id = 0, None
            for item_ref_id, ref_score in reference_clip_score.items():
                if f"segment_{i_id}" in item_ref_id:
                    thr = ref_score
                    if thr > ref_thr:
                        ref_thr = thr
                        ref_id = item_ref_id
            if ref_thr > c_score:
                self.target_tokens[f"token_{t_id}"] = [ref_id, ref_thr]

    def substitution(self):
        print('========================= SUBSTITUTION START ========================= ')
        self.T_hat = self.caption
        print(f'{self.T_hat=}')
        for target_token, ref in self.target_tokens.items():
            print(f'{target_token=} {ref=}')
            segment_info = ref[0]
            splitted = segment_info.split('_')
            segment_idx = splitted[1]

            token = self.candidate_tokens[target_token][0]
            g = self.candidate_tokens[target_token][2]
            if g == "f":
                reference_list = self.feature_reference.copy()
                feature_idx = int(splitted[3])
                substitute = reference_list[feature_idx]
            elif g == "obj":
                reference_list = self.object_reference.copy()
                object_idx = int(splitted[3])
                substitute = reference_list[object_idx]

            print(substitute)
            if self.generic:
                substitute = "unknown"
            self.T_hat = self.T_hat.replace(token, substitute)
            print(f'{self.T_hat=}')
        print('========================= SUBSTITUTION END =========================')

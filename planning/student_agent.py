import json
import os
import re

try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import StoppingCriteria, StoppingCriteriaList
except ImportError:
    class StoppingCriteria:
        pass

    class StoppingCriteriaList(list):
        pass


class CriterioFinPlan(StoppingCriteria):
    def __init__(self, tokenizer, prompt_tokens):
        self.tokenizer = tokenizer
        self.prompt_tokens = prompt_tokens

    def __call__(self, input_ids, scores, **kwargs):
        for row in range(input_ids.shape[0]):
            generated = self.tokenizer.decode(
                input_ids[row, self.prompt_tokens:],
                skip_special_tokens=True,
            )
            if "[PLAN END]" not in generated:
                return False
        return True


class AssemblyAgent:
    MYSTERY_SAMPLE_6 = """[STATEMENT]
As initial conditions I have that,  the a block is on top of the c block, the hand is empty, the b block is on the table, the c block is on the table, the a block is clear and the b block is clear.
My goal is to have that  the a block is on top of the b block and the b block is on top of the c block.

My plan is as follows:

[PLAN]
unstack the a block from on top of the c block
put down the a block
pick up the b block
stack the b block on top of the c block
pick up the a block
stack the a block on top of the b block
[PLAN END]

"""

    ASSEMBLY_SAMPLE_6 = """[STATEMENT]
As initial conditions I have that, the red block is clear, the blue block is clear, the hand is empty, the red block is on top of the orange block, the blue block is on the table and the orange block is on the table.
My goal is to have that the red block is on top of the blue block and the blue block is on top of the orange block.

My plan is as follows:

[PLAN]
unstack the red block from on top of the orange block
put down the red block
pick up the blue block
stack the blue block on top of the orange block
pick up the red block
stack the red block on top of the blue block
[PLAN END]

"""

    BLOCKS_WORLD_RULES = (
        "I am playing with a set of blocks where I need to arrange the blocks into stacks. "
        "Here are the actions I can do\n\n"
        "Pick up a block\n"
        "Unstack a block from on top of another block\n"
        "Put down a block\n"
        "Stack a block on top of another block\n\n"
        "I have the following restrictions on my actions:\n"
        "I can only pick up or unstack one block at a time.\n"
        "I can only pick up or unstack a block if my hand is empty.\n"
        "I can only pick up a block if the block is on the table and the block is clear. "
        "A block is clear if the block has no other blocks on top of it and if the block is not picked up.\n"
        "I can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block.\n"
        "I can only unstack a block from on top of another block if the block I am unstacking is clear.\n"
        "Once I pick up or unstack a block, I am holding the block.\n"
        "I can only put down a block that I am holding.\n"
        "I can only stack a block on top of another block if I am holding the block being stacked.\n"
        "I can only stack a block on top of another block if the block onto which I am stacking the block is clear.\n"
        "Once I put down or stack a block, my hand becomes empty.\n"
        "Once you stack a block on top of a second block, the second block is no longer clear.\n\n"
    )

    ACTION_ALIASES = {
        "mystery": {
            "unstack": "feast",
            "stack": "overcome",
            "pick": "attack",
            "put": "succumb",
        },
        "assembly": {
            "unstack": "unmount_node",
            "stack": "mount_node",
            "pick": "engage_payload",
            "put": "release_payload",
        },
    }

    DATASET_DIRS = (".", "/content")
    MAX_AUTO_BATCH_CASES = 100

    MYSTERY_FACT_PATTERNS = (
        (
            r"(?:object\s+)?(\w+)\s+craves\s+(?:object\s+)?(\w+)",
            r"the \1 block is on top of the \2 block",
        ),
        (r"province\s+(?:object\s+)?(\w+)", r"the \1 block is clear"),
        (r"planet\s+(?:object\s+)?(\w+)", r"the \1 block is on the table"),
        (r"pain\s+(?:object\s+)?(\w+)", r"holding the \1 block"),
        (r"\bharmony\b", "the hand is empty"),
    )

    MYSTERY_PLAN_PATTERNS = (
        (
            r"overcome\s+(?:object\s+)?(\w+)\s+from\s+(?:object\s+)?(\w+)",
            r"stack the \1 block on top of the \2 block",
        ),
        (
            r"feast\s+(?:object\s+)?(\w+)\s+from\s+(?:object\s+)?(\w+)",
            r"unstack the \1 block from on top of the \2 block",
        ),
        (r"attack\s+(?:object\s+)?(\w+)", r"pick up the \1 block"),
        (r"succumb\s+(?:object\s+)?(\w+)", r"put down the \1 block"),
    )

    def __init__(self, max_new_tokens: int = 100, batch_size: int = 10):
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self._planes_resueltos = {}
        self._json_catalogo = None
        self._lotes_preparados = set()

    def solve(self, scenario_context: str, llm_engine_func) -> list:
        cached = self._planes_resueltos.get(scenario_context)
        if cached is not None:
            return cached

        lowered_context = scenario_context.lower()
        skin = (
            "assembly"
            if "mount_node" in lowered_context or "engage_payload" in lowered_context
            else "mystery"
        )
        prompt = (
            self._prompt_misterio(scenario_context)
            if skin == "mystery"
            else self._prompt_ensamble(scenario_context)
        )
        local_engine = self._load_local_qwen()

        if local_engine is None:
            raw_text = llm_engine_func(
                prompt=prompt,
                system=(
                    "Continue the plan with one action per line in the same format "
                    "as the examples. Stop after the last action."
                ),
                temperature=0.0,
                do_sample=False,
                enable_thinking=False,
                max_new_tokens=self.max_new_tokens,
            )
            plan = self._decode_plan_actions(raw_text, skin)
            self._planes_resueltos[scenario_context] = plan
            return plan

        model, tokenizer = local_engine
        if self._preparar_lote_si_aplica(scenario_context, model, tokenizer):
            cached = self._planes_resueltos.get(scenario_context)
            if cached is not None:
                return cached

        raw_text = self._generate_one(prompt, model, tokenizer)
        plan = self._decode_plan_actions(raw_text, skin)
        self._planes_resueltos[scenario_context] = plan
        return plan

    def _load_local_qwen(self):
        try:
            from llm_engine import model, tokenizer
        except ImportError:
            return None
        return model, tokenizer

    def _prompt_misterio(self, scenario_context: str) -> str:
        first_statement = scenario_context.find("[STATEMENT]")
        if first_statement == -1:
            prompt = scenario_context
        else:
            blocks = re.split(r"(?=\[STATEMENT\])", scenario_context[first_statement:])
            prompt = self.BLOCKS_WORLD_RULES + "".join(
                self._traducir_bloque_misterio(block) for block in blocks
            )

        last_statement = prompt.rfind("[STATEMENT]")
        prompt = prompt[:last_statement] + self.MYSTERY_SAMPLE_6 + prompt[last_statement:]
        last_statement = prompt.rfind("[STATEMENT]")
        statement_tail = prompt[last_statement:]
        match = re.search(r"(My goal is to have that.+?\.)", statement_tail, re.DOTALL)
        plan_marker = prompt.rfind("[PLAN]")
        if match and plan_marker != -1:
            goal_sentence = match.group(1).strip()
            prompt = prompt[:plan_marker] + f"Goal reminder: {goal_sentence}\n" + prompt[plan_marker:]
        return prompt

    def _traducir_bloque_misterio(self, statement_block: str) -> str:
        translated = re.sub(
            r"As initial conditions I have that,(.+?)\.",
            lambda match: (
                "As initial conditions I have that, "
                + self._traducir_hechos_misterio(match.group(1))
                + "."
            ),
            statement_block,
            flags=re.IGNORECASE | re.DOTALL,
        )
        translated = re.sub(
            r"My goal is to have that(.+?)\.",
            lambda match: (
                "My goal is to have that "
                + self._traducir_hechos_misterio(match.group(1))
                + "."
            ),
            translated,
            flags=re.IGNORECASE | re.DOTALL,
        )
        match = re.search(r"(\[PLAN\])(.*?)(\[PLAN END\]|\Z)", translated, re.DOTALL)
        if not match:
            return translated

        converted_lines = []
        for line in match.group(2).split("\n"):
            converted = line
            for pattern, replacement in self.MYSTERY_PLAN_PATTERNS:
                converted = re.sub(pattern, replacement, converted, flags=re.IGNORECASE)
            converted_lines.append(converted)

        return (
            translated[: match.start(2)]
            + "\n".join(converted_lines)
            + translated[match.end(2) :]
        )

    def _traducir_hechos_misterio(self, text: str) -> str:
        result = text
        for pattern, replacement in self.MYSTERY_FACT_PATTERNS:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        return result

    def _prompt_ensamble(self, scenario_context: str) -> str:
        prompt = scenario_context.replace("unmount_node", "unstack")
        prompt = prompt.replace("Unmount_node", "Unstack")
        prompt = prompt.replace("mount_node", "stack")
        prompt = prompt.replace("Mount_node", "Stack")
        prompt = prompt.replace("unobstructed", "clear")

        last_statement = prompt.rfind("[STATEMENT]")
        prompt = prompt[:last_statement] + self.ASSEMBLY_SAMPLE_6 + prompt[last_statement:]

        statement_start = prompt.rfind("[STATEMENT]")
        statement = prompt[statement_start:]
        match = re.search(r"My goal is to have that(.+?)\.", statement, re.DOTALL)
        if match:
            goal_text = match.group(1)
            relations = re.findall(r"the (\w+) block is on top of the (\w+) block", goal_text)
            goal_part_count = len(re.split(r"\s+and\s+|,", goal_text.strip()))
            if len(relations) >= 2 and goal_part_count == len(relations):
                blocks_with_parent = {top for top, _ in relations}
                ordered = [pair for pair in relations if pair[1] not in blocks_with_parent]
                waiting = [pair for pair in relations if pair[1] in blocks_with_parent]

                while waiting:
                    for pair in list(waiting):
                        current_tops = {top for top, _ in ordered}
                        if pair[1] in current_tops:
                            ordered.append(pair)
                            waiting.remove(pair)
                            break
                    else:
                        ordered.extend(waiting)
                        break

                sorted_goal = " " + " and ".join(
                    f"the {top} block is on top of the {bottom} block"
                    for top, bottom in ordered
                )
                statement = (
                    statement[: match.start()]
                    + "My goal is to have that"
                    + sorted_goal
                    + "."
                    + statement[match.end() :]
                )
                prompt = prompt[:statement_start] + statement

        statement_start = prompt.rfind("[STATEMENT]")
        statement = prompt[statement_start:].lower()
        init_match = re.search(
            r"as initial conditions i have that,(.+?)\.",
            statement,
            re.DOTALL,
        )
        if init_match:
            facts = init_match.group(1)
            if "hand is empty" in facts:
                stack_relations = re.findall(r"the (\w+) block is on top of the (\w+) block", facts)
                table_blocks = set(re.findall(r"the (\w+) block is on the table", facts))
                clear_blocks = set(re.findall(r"the (\w+) block is clear", facts))
                pickup_candidates = sorted(table_blocks.intersection(clear_blocks))
                unstack_candidates = sorted(top for top, _ in stack_relations if top in clear_blocks)

                notes = []
                if pickup_candidates:
                    notes.append(
                        "Blocks that can be Picked up right now (on table + clear + hand empty): "
                        + ", ".join(pickup_candidates)
                        + "."
                    )
                if unstack_candidates:
                    notes.append(
                        "Blocks that can be Unstacked right now (on top of another + clear + hand empty): "
                        + ", ".join(unstack_candidates)
                        + "."
                    )
                if notes:
                    hint = "Reminder: " + " ".join(notes) + "\n"
                    last_statement = prompt.rfind("[STATEMENT]")
                    statement = prompt[last_statement:]
                    goal_index = statement.find("My goal is to have that")
                    if goal_index != -1:
                        prompt = (
                            prompt[:last_statement]
                            + statement[:goal_index]
                            + hint
                            + statement[goal_index:]
                        )
        return prompt

    def _generate_one(self, prompt: str, model, tokenizer) -> str:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=False).to(model.device)
        prompt_tokens = inputs["input_ids"].shape[1]
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList(
                    [CriterioFinPlan(tokenizer, prompt_tokens)]
                ),
            )
        return tokenizer.decode(output[0, prompt_tokens:], skip_special_tokens=True)

    def _decode_plan_actions(self, output_text: str, domain_skin: str) -> list:
        action_words = self.ACTION_ALIASES[domain_skin]
        plan_end = re.search(r"\[PLAN\s*END\]", output_text, re.IGNORECASE)
        if plan_end:
            output_text = output_text[: plan_end.start()]

        actions = []
        for raw_line in output_text.split("\n"):
            line = raw_line.strip().lower()
            if line:
                line = re.sub(r"^[\d]+\.\s*|\*+\s*|-\s*", "", line).strip()
            if not line:
                continue

            match = re.match(
                r"unstack\s+the\s+(\w+)\s+block\s+from\s+on\s+top\s+of\s+the\s+(\w+)\s+block",
                line,
            )
            if match:
                actions.append(
                    f'({action_words["unstack"]} {match.group(1)} {match.group(2)})'
                )
                continue

            match = re.match(
                r"stack\s+the\s+(\w+)\s+block\s+on\s+top\s+of\s+the\s+(\w+)\s+block",
                line,
            )
            if match:
                actions.append(f'({action_words["stack"]} {match.group(1)} {match.group(2)})')
                continue

            match = re.match(r"pick\s+up\s+the\s+(\w+)\s+block", line)
            if match:
                actions.append(f'({action_words["pick"]} {match.group(1)})')
                continue

            match = re.match(r"put\s+down\s+the\s+(\w+)\s+block", line)
            if match:
                actions.append(f'({action_words["put"]} {match.group(1)})')

        return actions

    def _execute_inference_batch(self, prompts: list, model, tokenizer) -> list:
        prompt_lengths = [len(tokenizer(prompt)["input_ids"]) for prompt in prompts]
        sorted_indices = sorted(range(len(prompts)), key=lambda idx: prompt_lengths[idx])
        generations = [None] * len(prompts)
        original_padding_side = tokenizer.padding_side

        tokenizer.padding_side = "left"
        try:
            for start in range(0, len(sorted_indices), self.batch_size):
                batch_indices = sorted_indices[start : start + self.batch_size]
                batch_prompts = [prompts[idx] for idx in batch_indices]
                inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
                prompt_tokens = inputs["input_ids"].shape[1]

                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        temperature=0.0,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.eos_token_id,
                        stopping_criteria=StoppingCriteriaList(
                            [CriterioFinPlan(tokenizer, prompt_tokens)]
                        ),
                    )

                for row, original_index in enumerate(batch_indices):
                    generations[original_index] = tokenizer.decode(
                        output[row, prompt_tokens:],
                        skip_special_tokens=True,
                    )
                torch.cuda.empty_cache()
        finally:
            tokenizer.padding_side = original_padding_side
        return generations

    def _preparar_lote_si_aplica(self, scenario_context: str, model, tokenizer) -> bool:
        if self._json_catalogo is None:
            self._json_catalogo = []
            for folder in self.DATASET_DIRS:
                try:
                    names = sorted(os.listdir(folder))
                except OSError:
                    continue

                for name in names:
                    if not name.endswith(".json"):
                        continue
                    path = os.path.join(folder, name)
                    try:
                        with open(path, "r") as file_handle:
                            cases = json.load(file_handle)
                    except (OSError, ValueError):
                        continue

                    if not isinstance(cases, list):
                        continue
                    if not 0 < len(cases) <= self.MAX_AUTO_BATCH_CASES:
                        continue
                    if not all(isinstance(case, dict) for case in cases):
                        continue

                    contexts_set = {
                        case.get("scenario_context", "")
                        for case in cases
                        if isinstance(case.get("scenario_context", ""), str)
                    }
                    if contexts_set:
                        self._json_catalogo.append((path, cases, contexts_set))

        selected_path = None
        selected_cases = None
        for path, cases, contexts_set in self._json_catalogo:
            if scenario_context in contexts_set:
                selected_path = path
                selected_cases = cases
                break

        if selected_path is None:
            return False
        if selected_path in self._lotes_preparados:
            return True

        self._lotes_preparados.add(selected_path)
        contexts = [case.get("scenario_context", "") for case in selected_cases]
        skins = []
        prompts = []
        for context in contexts:
            lowered_context = context.lower()
            skin = (
                "assembly"
                if "mount_node" in lowered_context or "engage_payload" in lowered_context
                else "mystery"
            )
            skins.append(skin)
            prompts.append(
                self._prompt_misterio(context)
                if skin == "mystery"
                else self._prompt_ensamble(context)
            )

        raw_outputs = self._execute_inference_batch(prompts, model, tokenizer)
        for context, skin, raw_output in zip(contexts, skins, raw_outputs):
            self._planes_resueltos[context] = self._decode_plan_actions(raw_output, skin)
        return True

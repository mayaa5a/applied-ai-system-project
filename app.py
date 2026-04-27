"""
app.py — PawPal+ Streamlit UI
Connects the Owner / Pet / Task / Scheduler logic layer to the browser UI.
"""

import streamlit as st
from datetime import date
from pathlib import Path

from pawpal_system import Owner, Pet, Task, Scheduler
from utils.rag import (
    EMERGENCY_MESSAGE,
    calculate_relevance_score,
    confidence_label,
    generate_ai_answer_with_error,
    is_emergency_question,
    load_knowledge_base,
    log_interaction,
    retrieve_context,
)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="PawPal+", page_icon="🐾", layout="wide")

# ── Session-state bootstrap ───────────────────────────────────────────────────
# Streamlit re-runs the entire script on every interaction.
# We store the Owner (and its pets/tasks) in st.session_state so data
# survives across re-runs without being reset to an empty object.

if "owner" not in st.session_state:
    st.session_state.owner = Owner("Jordan")
    st.session_state.scheduler = Scheduler(st.session_state.owner)

owner: Owner = st.session_state.owner
scheduler: Scheduler = st.session_state.scheduler

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🐾 PawPal+")
    st.caption("Pet care scheduling assistant")

    st.divider()

    # Owner name — update in-place so the Scheduler still references the same object
    new_name = st.text_input("Owner name", value=owner.name, key="owner_name_input")
    if new_name.strip() and new_name.strip() != owner.name:
        owner.name = new_name.strip()

    st.divider()
    st.subheader("Add a Pet")
    pet_name_input = st.text_input("Pet name", key="new_pet_name")
    species_input = st.selectbox(
        "Species", ["dog", "cat", "bird", "rabbit", "other"], key="new_pet_species"
    )
    if st.button("Add Pet", key="add_pet_btn"):
        if not pet_name_input.strip():
            st.warning("Enter a pet name first.")
        elif pet_name_input.strip() in [p.name for p in owner.get_pets()]:
            st.warning(f"'{pet_name_input.strip()}' already exists.")
        else:
            owner.add_pet(Pet(pet_name_input.strip(), species_input))
            st.success(f"Added {pet_name_input.strip()}!")
            st.rerun()

    # Pet roster summary
    pets = owner.get_pets()
    if pets:
        st.divider()
        st.subheader("Your Pets")
        for p in pets:
            task_count = len(p.get_tasks())
            done_count = sum(1 for t in p.get_tasks() if t.completed)
            st.markdown(
                f"**{p.name}** ({p.species})  \n"
                f"`{done_count}/{task_count}` tasks done"
            )
    else:
        st.info("No pets yet. Add one above.")

# ── Main area ─────────────────────────────────────────────────────────────────

st.title(f"🐾 Welcome, {owner.name}!")

st.header("Ask PawPal AI")
st.caption("Ask a pet-care question and PawPal will answer using its local guide.")

# The knowledge base lives in the project so the app can run without a database.
knowledge_base_path = Path("data/pet_care_knowledge_base.txt")
knowledge_text = load_knowledge_base(knowledge_base_path)
if not knowledge_text:
    st.warning(
        "PawPal could not find the local pet-care knowledge base. "
        "Answers will use a cautious fallback until the file is restored."
    )

user_question = st.text_area(
    "Pet-care question",
    placeholder="Example: How often should I walk my dog?",
    key="pawpal_ai_question",
)

if st.button("Ask PawPal", key="ask_pawpal_btn"):
    clean_question = user_question.strip()
    if not clean_question:
        st.warning("Enter a question first.")
    else:
        retrieved_context = retrieve_context(clean_question, knowledge_text, top_k=2)
        confidence_score = calculate_relevance_score(clean_question, retrieved_context)
        emergency_triggered = is_emergency_question(clean_question)
        error_message = ""

        # Emergency questions should never call the AI model.
        if emergency_triggered:
            answer = EMERGENCY_MESSAGE
            st.error(answer)
        else:
            answer, error_message = generate_ai_answer_with_error(
                clean_question,
                retrieved_context,
            )
            st.success("PawPal AI answer")
            if error_message:
                st.warning(
                    "PawPal could not reach the AI model, so it used the retrieved "
                    "pet-care guide to create a fallback answer."
                )

        log_interaction(
            clean_question,
            retrieved_context,
            answer,
            confidence_score=confidence_score,
            emergency_triggered=emergency_triggered,
            error_message=error_message,
        )

        st.subheader("Reliability score")
        st.write(f"{confidence_score:.2f} ({confidence_label(confidence_score)})")
        if confidence_score < 0.5:
            st.warning(
                "PawPal found limited matching information, so this answer may be "
                "less reliable. Consider checking with a vet or trusted pet-care source."
            )

        st.subheader("Retrieved context")
        if retrieved_context:
            st.info(retrieved_context)
        else:
            st.info("No matching guide section was retrieved.")

        st.subheader("Final answer")
        st.write(answer)

st.divider()

pets = owner.get_pets()

if not pets:
    st.info("👈 Add a pet in the sidebar to get started.")
    st.stop()

tab_schedule, tab_add, tab_filter = st.tabs(
    ["📅 Today's Schedule", "➕ Add Tasks", "🔍 Filter Tasks"]
)

# ── Tab 1: Today's Schedule ───────────────────────────────────────────────────

with tab_schedule:
    today_str = date.today().strftime("%A, %B %d, %Y")
    st.subheader(f"Schedule — {today_str}")

    schedule = scheduler.get_schedule()
    conflicts = scheduler.get_conflicts()

    # Conflict banner
    if conflicts:
        for name_a, task_a, name_b, task_b in conflicts:
            st.warning(
                f"⚠️ **Conflict:** {name_a}'s *{task_a.description}* "
                f"({task_a.time}, {task_a.duration} min) overlaps with "
                f"{name_b}'s *{task_b.description}* "
                f"({task_b.time}, {task_b.duration} min)."
            )
    else:
        if schedule:
            st.success("✅ No scheduling conflicts detected.")

    st.divider()

    if not schedule:
        st.info("No tasks scheduled yet. Go to **Add Tasks** to create some.")
    else:
        for pet_name, task in schedule:
            col_icon, col_info, col_action = st.columns([0.5, 7, 2])
            with col_icon:
                st.markdown("✅" if task.completed else "⬜")
            with col_info:
                due_label = f" · due {task.due_date}" if task.due_date else ""
                st.markdown(
                    f"**{task.time}** &nbsp;|&nbsp; **{pet_name}** &nbsp;—&nbsp; "
                    f"{task.description} &nbsp;·&nbsp; {task.duration} min "
                    f"· *{task.frequency}*{due_label}"
                )
            with col_action:
                btn_key = f"done_{pet_name}_{task.description}_{task.time}_{task.due_date}"
                if task.completed:
                    st.caption("done")
                else:
                    if st.button("Mark done", key=btn_key):
                        scheduler.mark_task_complete(pet_name, task.description)
                        st.rerun()

        # Progress bar
        total = len(schedule)
        done = sum(1 for _, t in schedule if t.completed)
        st.divider()
        st.progress(
            done / total if total else 0,
            text=f"{done} of {total} tasks completed today",
        )

# ── Tab 2: Add Tasks ──────────────────────────────────────────────────────────

with tab_add:
    st.subheader("Add a Task")

    pet_options = [p.name for p in pets]
    selected_pet_name = st.selectbox("Assign to pet", pet_options, key="task_pet_select")

    col_left, col_right = st.columns(2)
    with col_left:
        task_desc = st.text_input("Task description", value="Morning walk", key="t_desc")
        task_time = st.text_input("Time (HH:MM, 24-hour)", value="07:00", key="t_time")
    with col_right:
        task_dur = st.number_input(
            "Duration (minutes)", min_value=1, max_value=480, value=30, key="t_dur"
        )
        task_freq = st.selectbox(
            "Frequency", ["daily", "weekly", "once"], key="t_freq"
        )

    if st.button("Add Task", key="add_task_btn"):
        # Validate time format
        try:
            h_str, m_str = task_time.strip().split(":")
            assert 0 <= int(h_str) <= 23 and 0 <= int(m_str) <= 59
            valid_time = True
        except Exception:
            valid_time = False

        if not task_desc.strip():
            st.error("Task description cannot be empty.")
        elif not valid_time:
            st.error("Time must be in HH:MM format (e.g. 08:30).")
        else:
            target_pet = next(p for p in pets if p.name == selected_pet_name)
            target_pet.add_task(
                Task(
                    description=task_desc.strip(),
                    time=task_time.strip(),
                    duration=int(task_dur),
                    frequency=task_freq,
                )
            )
            st.success(f"Added '{task_desc.strip()}' to {selected_pet_name}.")
            st.rerun()

    # Per-pet task overview
    st.divider()
    st.subheader("Current Tasks by Pet")
    for pet in pets:
        tasks = pet.get_tasks()
        if tasks:
            with st.expander(f"{pet.name} ({pet.species}) — {len(tasks)} task(s)", expanded=True):
                rows = [
                    {
                        "Time": t.time,
                        "Task": t.description,
                        "Duration (min)": t.duration,
                        "Frequency": t.frequency,
                        "Status": "✅ Done" if t.completed else "⬜ Pending",
                    }
                    for t in sorted(tasks, key=lambda x: x.time)
                ]
                st.table(rows)
        else:
            st.markdown(f"**{pet.name}** — no tasks yet.")

# ── Tab 3: Filter Tasks ───────────────────────────────────────────────────────

with tab_filter:
    st.subheader("Filter Tasks")

    col_fa, col_fb = st.columns(2)
    with col_fa:
        filter_pet = st.selectbox(
            "Filter by pet", ["All"] + [p.name for p in pets], key="f_pet"
        )
    with col_fb:
        filter_status = st.selectbox(
            "Filter by status", ["All", "Pending", "Completed"], key="f_status"
        )

    pet_arg = None if filter_pet == "All" else filter_pet
    completed_arg: bool | None = (
        None if filter_status == "All" else filter_status == "Completed"
    )

    filtered = scheduler.filter_tasks(pet_name=pet_arg, completed=completed_arg)

    if not filtered:
        st.info("No tasks match the selected filters.")
    else:
        rows = [
            {
                "Pet": name,
                "Time": task.time,
                "Task": task.description,
                "Duration (min)": task.duration,
                "Frequency": task.frequency,
                "Due": task.due_date or "today",
                "Status": "✅ Done" if task.completed else "⬜ Pending",
            }
            for name, task in filtered
        ]
        st.table(rows)
        st.caption(f"{len(rows)} task(s) shown")

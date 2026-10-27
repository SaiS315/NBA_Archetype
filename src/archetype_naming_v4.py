"""V4 contribution-aware, evidence-gated basketball archetype names.
Run from project root: python src/archetype_naming_v4.py
Uses V3 outputs and the same player data; never overwrites V3.
Scores describe STATISTICAL TENDENCIES, not player impact or defensive quality.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data/processed/player_two_way_profiles_tracking.csv'
V3 = ROOT / 'results/archetype_naming_v3/archetype_names_v3.csv'
OUTPUT = ROOT / 'results/archetype_naming_v4'

FEATURES = [
    'PTS_PER36','FG3A_PER36','FG2A_PER36','FTA_PER36','TS_PCT','OREB_PER36',
    'AST_PCT','AST_TO','TO_RATIO','DEFLECTIONS_PER36','CONTESTED_2PT_PER36',
    'CONTESTED_3PT_PER36','DREB_PCT','DEF_LOOSE_PER36','DEF_BOXOUTS_PER36',
]

# Explicit, adjustable HEURISTICS, not trained classification thresholds.
STRONG = 75
SUPPORTING = 65
EXCEPTIONAL = 90

def role_pos(position):
    parts = set(str(position).split('-'))
    if 'Center' in parts:
        return 'big'
    if parts == {'Guard'}:
        return 'guard'
    if parts == {'Forward'}:
        return 'forward'
    return 'wing'


def label_player(r):
    """Assign archetypes using supported statistical roles."""
    p = lambda name: float(r["P_" + name])

    position = role_pos(r["POSITION"])
    big = position == "big"
    forward = position == "forward"

    scoring = float(r["SCORING_ROLE_SCORE"])
    passing = float(r["PLAYMAKING_ROLE_SCORE"])
    defense = float(r["DEFENSIVE_ACTIVITY_SCORE"])

    pts = p("PTS_PER36")
    threes = p("FG3A_PER36")
    twos = p("FG2A_PER36")
    assists = p("AST_PCT")
    contests2 = p("CONTESTED_2PT_PER36")
    contests3 = p("CONTESTED_3PT_PER36")
    rebounds = p("DREB_PCT")
    deflections = p("DEFLECTIONS_PER36")
    loose = p("DEF_LOOSE_PER36")

    # 1. Identify scoring characteristics.
    volume = pts >= 75
    perimeter = threes >= 80

    interior_finisher = (
        p("OREB_PER36") >= 75
        and p("TS_PCT") >= 60
        and threes <= 55
    )

    # 2. Identify playmaking responsibilities.
    # High AST% alone doesn't make a scorer a playmaker.
    big_hub = (
        big
        and assists >= 88
        and passing >= 70
        and (volume or twos >= 70)
    )

    primary_creator = (
        assists >= 90
        and passing >= 75
        and (not volume or assists >= 95)
    )

    secondary_creator = (
        assists >= 72
        and passing >= 63
    )

    # 3. Identify defensive specialization.
    disruptor = (
        deflections >= 85
        and (loose >= 60 or deflections >= 95)
    )

    interior_defender = (
        big
        and defense >= 55
        and contests2 >= 90
        and (rebounds >= 60 or contests2 >= 95)
    )

    interior_activity = (
        contests2 >= 80
        and rebounds >= 70
    )

    perimeter_contest = (
        contests3 >= 85
        and defense >= 60
    )

    defensive_specialist = (
        (interior_defender or disruptor or perimeter_contest)
        and defense >= 55
    )

    # 4. Identify meaningful hybrid roles.
    defensive_playmaker = (
        passing >= 68
        and defense >= 68
        and assists >= 80
        and defense >= passing - 8
        and (
            contests2 >= 75
            or deflections >= 75
            or rebounds >= 85
        )
    )

    scoring_playmaker = (
        volume
        and passing >= 75
        and assists >= 90
        and (
            assists >= 95
            or passing >= scoring + 10
        )
    )

    styles = []

    if (
        (volume and scoring >= 60)
        or perimeter
        or interior_finisher
    ):
        styles.append("scoring")

    if big_hub or primary_creator or secondary_creator:
        styles.append("playmaking")

    if defensive_specialist or (
        defense >= 68 and interior_activity
    ):
        styles.append("defensive activity")

    hybrid = (
        defensive_playmaker
        or scoring_playmaker
        or (big_hub and volume)
    )

    # 5. Build review reasons.
    reasons = []

    if not styles:
        reasons.append(
            "No strongly supported specialized role"
        )

    if hybrid:
        reasons.append(
            "Multiple meaningful role dimensions"
        )

    scores = sorted([scoring, passing, defense])

    if scores[-1] - scores[-2] < 6:
        reasons.append(
            "Close dimension scores; inspect primary priority"
        )

    # 6. Generate the primary archetype.
    # Rule order determines priority.
    if big_hub:
        label = "Playmaking Offensive Hub"

    elif defensive_playmaker:
        if big:
            label = "Defensive Playmaking Big"
        elif forward:
            label = "Defensive Playmaking Forward"
        else:
            label = "Defensive Playmaking Connector"

    elif interior_defender:
        label = "Interior Defensive Specialist"

    elif scoring_playmaker:
        label = "Scoring Playmaker"

    elif disruptor and defense >= 55 and not volume:
        label = "Ball-Disrupting Connector"

    elif volume and perimeter and forward and twos >= 60:
        label = "Versatile Scoring Forward"

    elif volume and perimeter:
        label = "Perimeter Scoring Creator"

    elif volume and twos >= 75:
        label = "High-Volume Inside-Arc Scorer"

    elif volume:
        label = "Scoring Creator"

    elif primary_creator:
        label = "Primary Playmaker"

    elif perimeter:
        label = "Perimeter Shooting Specialist"

    elif interior_finisher and big:
        label = "Interior Finisher and Rebounder"

    elif disruptor:
        label = "Ball-Disrupting Contributor"

    elif interior_activity and defense >= 60:
        label = "Interior Contest and Rebounding Contributor"

    elif secondary_creator:
        label = "Secondary Playmaker"

    elif perimeter_contest:
        label = "Perimeter Contest Contributor"

    elif interior_finisher:
        label = "Interior Finishing Contributor"

    else:
        label = "Supporting Role Player"
        reasons.append(
            "Generic label: limited distinctive evidence"
        )

    if (
        "Defensive" in label
        or "Ball-Disrupting" in label
        or "Contest" in label
    ):
        reasons.append(
            "Defensive activity is not proven defensive impact"
        )

    # 7. Preserve separate dimensional identities.
    if volume and perimeter:
        scoring_identity = "High-volume perimeter scorer"
    elif volume and twos >= 75:
        scoring_identity = "High-volume inside-arc scorer"
    elif volume:
        scoring_identity = "High-volume scorer"
    elif perimeter:
        scoring_identity = "Perimeter shooting specialist"
    elif interior_finisher:
        scoring_identity = "Interior finishing and rebounding"
    else:
        scoring_identity = "Complementary scoring profile"

    if big_hub:
        playmaking_identity = "High-involvement offensive hub"
    elif primary_creator:
        playmaking_identity = "Primary playmaking involvement"
    elif secondary_creator:
        playmaking_identity = "Secondary facilitation"
    else:
        playmaking_identity = "Lower recorded assist involvement"

    if interior_defender:
        defensive_identity = "Interior contest and rebounding specialist"
    elif disruptor:
        defensive_identity = "High ball-disruption activity"
    elif interior_activity:
        defensive_identity = "Interior contests and rebounding"
    elif perimeter_contest:
        defensive_identity = "High perimeter contest volume"
    else:
        defensive_identity = "Mixed recorded defensive activity"

    # 8. Generate supporting traits and evidence.
    traits = []

    for active, description in [
        (perimeter, "3-point attempt volume"),
        (volume, "scoring volume"),
        (big_hub or primary_creator, "high assist involvement"),
        (
            secondary_creator and not primary_creator,
            "secondary facilitation"
        ),
        (interior_finisher, "interior finishing/rebounding"),
        (interior_defender, "interior defensive contests"),
        (disruptor, "deflections"),
        (perimeter_contest, "perimeter shot contests"),
    ]:
        if active:
            traits.append(description)

    evidence = (
        f"PTS/36 percentile {pts:.0f}; "
        f"AST% percentile {assists:.0f}; "
        f"3PA/36 percentile {threes:.0f}; "
        f"deflections/36 percentile {deflections:.0f}; "
        f"2PT contests/36 percentile {contests2:.0f}; "
        f"defensive rebounds% percentile {rebounds:.0f}"
    )

    return dict(
        PRIMARY_ARCHETYPE=label,
        SCORING_IDENTITY=scoring_identity,
        PLAYMAKING_IDENTITY=playmaking_identity,
        DEFENSIVE_IDENTITY=defensive_identity,
        SECONDARY_TRAITS=(
            "; ".join(traits)
            if traits else "No dominant measured specialty"
        ),
        IS_HYBRID=hybrid,
        SUPPORTED_DIMENSIONS=(
            "; ".join(styles) if styles else "none"
        ),
        SUPPORTING_EVIDENCE=evidence,
        REVIEW_STATUS="Review" if reasons else "Provisional",
        REVIEW_REASON="; ".join(dict.fromkeys(reasons)),
    )


def main():
    stats = pd.read_csv(DATA)
    v3 = pd.read_csv(V3)
    for name, data in [('statistics',stats),('V3',v3)]:
        if not data['PLAYER_ID'].is_unique:
            raise ValueError(f'Duplicate player IDs in {name}')
    required_v3 = ['PLAYER_ID','PLAYER_NAME','POSITION','OFF_CLUSTER','DEF_CLUSTER',
                   'PRIMARY_ARCHETYPE','SCORING_ROLE_SCORE','PLAYMAKING_ROLE_SCORE',
                   'DEFENSIVE_ACTIVITY_SCORE']
    missing = [c for c in required_v3 if c not in v3] + [c for c in FEATURES if c not in stats]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    if set(v3.PLAYER_ID) != set(stats.PLAYER_ID):
        raise ValueError('V3 and tracking player populations differ; reconcile before naming')
    cols = ['PLAYER_ID'] + FEATURES
    merged = v3[required_v3].merge(stats[cols], on='PLAYER_ID', validate='one_to_one')
    for f in FEATURES:
        merged[f] = pd.to_numeric(merged[f],errors='coerce')
    if not np.isfinite(merged[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError('Missing or non-finite statistics')
    # All percentiles use the SAME eligible 378-player cohort.
    for f in FEATURES:
        merged['P_' + f] = merged[f].rank(pct=True,method='average') * 100
    merged['P_TO_RATIO'] = 100 - merged['P_TO_RATIO']

    generated = pd.DataFrame([label_player(row) for _,row in merged.iterrows()])
    out = pd.concat([merged[['PLAYER_ID','PLAYER_NAME','POSITION','OFF_CLUSTER','DEF_CLUSTER',
                             'SCORING_ROLE_SCORE','PLAYMAKING_ROLE_SCORE','DEFENSIVE_ACTIVITY_SCORE']].reset_index(drop=True),
                     generated],axis=1)
    compare = out[['PLAYER_ID','PLAYER_NAME','PRIMARY_ARCHETYPE','REVIEW_STATUS','REVIEW_REASON']].merge(
        v3[['PLAYER_ID','PRIMARY_ARCHETYPE']].rename(columns={'PRIMARY_ARCHETYPE':'V3_NAME'}),
        on='PLAYER_ID',validate='one_to_one')
    compare = compare.rename(columns={'PRIMARY_ARCHETYPE':'V4_NAME'})
    compare['NAME_CHANGED'] = compare.V3_NAME != compare.V4_NAME
    dist = out.PRIMARY_ARCHETYPE.value_counts().rename_axis('PRIMARY_ARCHETYPE').reset_index(name='PLAYER_COUNT')
    OUTPUT.mkdir(parents=True,exist_ok=True)
    out.to_csv(OUTPUT/'archetype_names_v4.csv',index=False)
    dist.to_csv(OUTPUT/'archetype_distribution_v4.csv',index=False)
    compare.to_csv(OUTPUT/'v3_v4_comparison.csv',index=False)
    out[out.REVIEW_STATUS=='Review'].to_csv(OUTPUT/'naming_review_v4.csv',index=False)
    print('Players:',len(out),'Names changed:',int(compare.NAME_CHANGED.sum()),
          'Review flags:',int((out.REVIEW_STATUS=='Review').sum()))
    print('\nArchetype distribution:\n',dist.to_string(index=False))
    for pid in [203999,1630578,201950,1627936,203110]:
        r=out[out.PLAYER_ID==pid]
        if not r.empty:
            print('\n',r[['PLAYER_NAME','PRIMARY_ARCHETYPE','SCORING_IDENTITY',
                           'PLAYMAKING_IDENTITY','DEFENSIVE_IDENTITY']].to_string(index=False))
    print('\nSaved to:',OUTPUT)

if __name__ == '__main__':
    main()

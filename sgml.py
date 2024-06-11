import re
markup = """
<h2 class="debateHeaderProp">This house believes that society benefits when we share personal information online.</h2>
<span class="debateFormat">Oregon-Oxford, Cross Examination</span>
<div class="debateAffirmSide">On the affirmative: Foo Debate Club</div>
<div class="debateOpposeSide">On the opposition: Bar Debate Club</div>
"""

print(re.sub('<[^>]*>', '',markup))

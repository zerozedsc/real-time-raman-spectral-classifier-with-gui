17-9-2024 0250
We have these problems.

1. Still problems with preprocess step inside pipeline vanish when i jump from preprocessed dataset (i enable some step), it shows in next raw dataset i click, but vanished in the other next. Im think for this, maybe we should save it in memory like list or dict like how we do our current pipeline right? So those step will not disappear if we not remove it or replace with steps from preprocessed dataset. Thats why for problem like steps different, we can compare steps inside our current memory and steps that been saved within that preprocessed dataset.
2. Still no padding for focused spectrum, Please check again why there is no padding on X, padding I want is like how it shows for raw data without any preprocessing did on that graph
3. Preview off still shows, nothing in graph. It should show current selected dataset graph without preprocessing inside pipeline section being implimented, but when i click manual focus, the graph will be shown but in focused without padding
4. Manual preview button still not working well, right now i dont know what will this button do
5. I dont like the orange-like colour used here for font inside preprocess section, maybe can stay with black and gray. Also I think, if we press disable/enable toggle in steps when current dataset is raw dataset, it should also change font colour like how we do with preprocessed data (black for enablem gray for disable)

Dont forget to always impliment localization as this was important component, you can look in the code how we impliment localization. You also can update locales files

After fix, adjust and update the implimentation. Update related .md docs and .AGENT-AI

Lastly clean all Debug code and debug related print, debug related console.log and debug related create logs

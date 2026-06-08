// Chapter: 具体的优化主题
// Example 14.7a. Testing multiple conditions

enum Weekdays {
    Sunday, Monday, Tuesday, Wednesday, Thursday, Friday, Saturday
};
Weekdays Day;
if (Day == Tuesday || Day == Wednesday || Day == Friday)
{
    DoThisThreeTimesAWeek();
}

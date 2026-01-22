const mongoose = require("mongoose");
const bcrypt = require("bcryptjs");
const Employee = require("./models/Employee");

mongoose.connect(
  "mongodb+srv://2324046_db_user:kaviya2602@cluster0.lamp0vs.mongodb.net/wfh_attendance?retryWrites=true&w=majority"
)
.then(async () => {

    const password = "kaka234"; // plaintext password
    const hashed = await bcrypt.hash(password, 10);

    const emp = new Employee({
        employeeId: "EMP-1234",
        password: hashed,
        name: "Kaviya",
        email: "kaviya@example.com",
        salary: 50000
    });

    await emp.save();
    console.log("Employee created successfully");
    process.exit();

})
.catch(err => console.error(err));

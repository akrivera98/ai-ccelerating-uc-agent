using JuMP
using HiGHS
using UnitCommitment

function run_ED(filename)
    ex_instance = UnitCommitment.read(filename)
    solver = HiGHS.Optimizer

    # 1. Construct optimization model
    model = UnitCommitment.build_model(
        instance = ex_instance,
        optimizer = solver,
    )

    set_attribute(model, "output_flag", false)
    set_attribute(model, "mip_rel_gap", 1e-6)
    
    # 2. Solve model
    UnitCommitment.optimize!(model)
    
    # UnitCommitment.write("OutputData.json", solution)

    if termination_status(model) != MOI.OPTIMAL
        value_cost = 1e9
        status = "infeasible"
        solution = []
    else
        value_cost = objective_value(model)
        status = "optimal"
        solution = UnitCommitment.solution(model)
    end

    return round(solve_time(model),digits=2), value_cost, status, solution

end
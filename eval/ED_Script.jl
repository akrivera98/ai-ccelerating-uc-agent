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
    # Name the UC binaries so IIS becomes interpretable
    # for (idx, v) in model[:is_on]
    #     set_name(v, "is_on[$idx]")
    # end
    # for (idx, v) in model[:switch_on]
    #     set_name(v, "switch_on[$idx]")
    # end
    # for (idx, v) in model[:switch_off]
    #     set_name(v, "switch_off[$idx]")
    # end
    # write_to_file(model, "full_before_solve.lp")
    # 2. Solve model
    UnitCommitment.optimize!(model)
    
    # UnitCommitment.write("OutputData.json", solution)

    if termination_status(model) != MOI.OPTIMAL
        value_cost = 1e9
        status = "infeasible"
        # MOI.compute_conflict!(backend(model))
        # println("ConflictStatus: ", MOI.get(backend(model), MOI.ConflictStatus()))
        # # Copy the IIS into a new model you can inspect/export
        # iis_model, _ = copy_conflict(model)
        # write_to_file(iis_model, "uc_iis.lp")
        solution = []
    else
        value_cost = objective_value(model)
        status = "optimal"
        solution = UnitCommitment.solution(model)
    end

    return round(solve_time(model),digits=2), value_cost, status, solution

end
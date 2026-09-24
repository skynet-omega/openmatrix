#include "epoch_session.hpp"
#include "mujoco_owner.hpp"

#include <mujoco/mujoco.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using epoch::Context;
using epoch::Owner;
using epoch::Port;
using epoch::Session;

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

class PulseOwner final : public Owner {
 public:
  std::string name() const override { return "pulse"; }
  std::vector<Port> reads() const override { return {}; }
  std::vector<Port> writes() const override { return {{"sensory", "pulse"}}; }
  void snapshot() override { saved_ = calls_; }
  void advance(Context& context) override {
    ++calls_;
    context.publish(context.now_ns + 500000, "sensory", 1.0);
  }
  void commit() override {}
  void rollback() noexcept override { calls_ = saved_; }
  void release_snapshot() noexcept override {}
  int calls() const { return calls_; }

 private:
  int calls_ = 0;
  int saved_ = 0;
};

class NeuronOwner final : public Owner {
 public:
  std::string name() const override { return "synthetic_neuron"; }
  std::vector<Port> reads() const override { return {{"sensory", "pulse"}}; }
  std::vector<Port> writes() const override { return {{"motor", "motor_ctrl"}}; }
  void snapshot() override { saved_ = activation_; }
  void advance(Context& context) override {
    const auto t = context.now_ns;
    if (context.sum_events("sensory", t + 250000) != 0.0) {
      throw std::runtime_error("future event leaked into past");
    }
    activation_ += context.sum_events("sensory", t + 500000);
    context.set_value("motor", activation_);
  }
  void commit() override {}
  void rollback() noexcept override { activation_ = saved_; }
  void release_snapshot() noexcept override {}
  double activation() const { return activation_; }

 private:
  double activation_ = 0.0;
  double saved_ = 0.0;
};

class FaultOwner final : public Owner {
 public:
  explicit FaultOwner(bool fail) : fail_(fail) {}
  std::string name() const override { return "late_fault"; }
  std::vector<Port> reads() const override { return {}; }
  std::vector<Port> writes() const override { return {}; }
  void snapshot() override {}
  void advance(Context&) override {
    if (fail_) throw std::runtime_error("injected late failure");
  }
  void commit() override {}
  void rollback() noexcept override {}
  void release_snapshot() noexcept override {}
  void set_fail(bool fail) { fail_ = fail; }

 private:
  bool fail_;
};

class BadOwner final : public Owner {
 public:
  BadOwner(std::string owner_name, Port read, Port write)
      : name_(std::move(owner_name)), read_(std::move(read)), write_(std::move(write)) {}
  std::string name() const override { return name_; }
  std::vector<Port> reads() const override { return read_.name.empty() ? std::vector<Port>{} : std::vector<Port>{read_}; }
  std::vector<Port> writes() const override { return write_.name.empty() ? std::vector<Port>{} : std::vector<Port>{write_}; }
  void snapshot() override {}
  void advance(Context&) override {}
  void commit() override {}
  void rollback() noexcept override {}
  void release_snapshot() noexcept override {}

 private:
  std::string name_;
  Port read_;
  Port write_;
};

std::vector<mjtNum> integration_state(mjModel* model, mjData* data) {
  const int count = mj_stateSize(model, mjSTATE_INTEGRATION);
  std::vector<mjtNum> state(count);
  mj_getState(model, data, state.data(), mjSTATE_INTEGRATION);
  return state;
}

struct Built {
  std::unique_ptr<Session> session;
  PulseOwner* pulse;
  NeuronOwner* neuron;
  FaultOwner* fault;
};

Built make_session(mjModel* model, mjData* data, bool fail) {
  auto session = std::make_unique<Session>(1000000);
  auto pulse = std::make_unique<PulseOwner>();
  auto neuron = std::make_unique<NeuronOwner>();
  auto fault = std::make_unique<FaultOwner>(fail);
  auto* pulse_ptr = pulse.get();
  auto* neuron_ptr = neuron.get();
  auto* fault_ptr = fault.get();
  session->add(std::move(pulse));
  session->add(std::move(neuron));
  session->add(std::make_unique<epoch::MuJoCoBodyOwner>(model, data, "motor"));
  session->add(std::move(fault));
  session->compile();
  return {std::move(session), pulse_ptr, neuron_ptr, fault_ptr};
}

}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 2, "expected fixture XML path");
    char error[1024] = {};
    std::unique_ptr<mjModel, decltype(&mj_deleteModel)> model(mj_loadXML(argv[1], nullptr, error, sizeof(error)), mj_deleteModel);
    if (!model) throw std::runtime_error(std::string("MuJoCo XML: ") + error);
    std::unique_ptr<mjData, decltype(&mj_deleteData)> trial_data(mj_makeData(model.get()), mj_deleteData);
    std::unique_ptr<mjData, decltype(&mj_deleteData)> control_data(mj_makeData(model.get()), mj_deleteData);
    require(trial_data && control_data, "MuJoCo data allocation failed");

    bool duplicate_rejected = false;
    try {
      Session bad(1000000);
      bad.add(std::make_unique<BadOwner>("writer_a", Port{}, Port{"x", "pulse"}));
      bad.add(std::make_unique<BadOwner>("writer_b", Port{}, Port{"x", "pulse"}));
      bad.compile();
    } catch (const std::runtime_error&) { duplicate_rejected = true; }
    require(duplicate_rejected, "duplicate writer accepted");

    bool unit_rejected = false;
    try {
      Session bad(1000000);
      bad.add(std::make_unique<BadOwner>("writer", Port{}, Port{"x", "pulse"}));
      bad.add(std::make_unique<BadOwner>("reader", Port{"x", "motor_ctrl"}, Port{}));
      bad.compile();
    } catch (const std::runtime_error&) { unit_rejected = true; }
    require(unit_rejected, "unit mismatch accepted");

    auto trial = make_session(model.get(), trial_data.get(), true);
    const auto before = integration_state(model.get(), trial_data.get());
    bool late_error = false;
    try { trial.session->tick(); }
    catch (const std::runtime_error& e) { late_error = std::string(e.what()) == "injected late failure"; }
    require(late_error, "late failure did not occur after MuJoCo step");
    require(before == integration_state(model.get(), trial_data.get()), "MuJoCo integration state not restored");
    require(trial.pulse->calls() == 0 && trial.neuron->activation() == 0.0, "neural owners not restored");
    require(trial.session->time_ns() == 0 && trial.session->context().events().empty(), "context not restored");

    trial.fault->set_fail(false);
    trial.session->tick();
    auto control = make_session(model.get(), control_data.get(), false);
    control.session->tick();
    require(integration_state(model.get(), trial_data.get()) == integration_state(model.get(), control_data.get()),
            "recovered tick differs from clean control");
    require(trial.pulse->calls() == control.pulse->calls(), "pulse owner differs after rollback");
    require(trial.neuron->activation() == control.neuron->activation(), "neuron owner differs after rollback");
    require(trial.session->time_ns() == control.session->time_ns(), "session time differs after rollback");
    require(trial.session->context().events().size() == control.session->context().events().size(), "event count differs");
    require(trial_data->qpos[0] > 0.0, "body did not move in accepted tick");
    std::cout << "PASS duplicate=1 unit=1 no_early_event=1 rollback=1 clean_replay=1"
              << " qpos=" << trial_data->qpos[0] << " neuron=" << trial.neuron->activation() << '\n';
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL " << e.what() << '\n';
    return 1;
  }
}

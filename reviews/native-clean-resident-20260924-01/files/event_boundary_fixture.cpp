#include <epoch_session.hpp>

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

void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}

class Publisher final : public Owner {
 public:
  Publisher(double at_end_first, double at_start_second)
      : at_end_first_(at_end_first), at_start_second_(at_start_second) {}
  std::string name() const override { return "publisher"; }
  std::vector<Port> reads() const override { return {}; }
  std::vector<Port> writes() const override { return {{"pulse", "count"}}; }
  void snapshot() override { saved_ = calls_; }
  void advance(Context& context) override {
    if (calls_ == 0 && at_end_first_ != 0.0) {
      context.publish(context.end_ns, "pulse", at_end_first_);
    }
    if (calls_ == 1 && at_start_second_ != 0.0) {
      context.publish(context.now_ns, "pulse", at_start_second_);
    }
    ++calls_;
  }
  void commit() override {}
  void rollback() noexcept override { calls_ = saved_; }
  void release_snapshot() noexcept override {}
  int calls() const { return calls_; }

 private:
  double at_end_first_;
  double at_start_second_;
  int calls_ = 0;
  int saved_ = 0;
};

class Receiver final : public Owner {
 public:
  std::string name() const override { return "receiver"; }
  std::vector<Port> reads() const override { return {{"pulse", "count"}}; }
  std::vector<Port> writes() const override { return {}; }
  void snapshot() override { saved_ = total_; }
  void advance(Context& context) override {
    total_ += context.sum_events("pulse", context.end_ns);
  }
  void commit() override {}
  void rollback() noexcept override { total_ = saved_; }
  void release_snapshot() noexcept override {}
  double total() const { return total_; }

 private:
  double total_ = 0.0;
  double saved_ = 0.0;
};

class LateFailure final : public Owner {
 public:
  std::string name() const override { return "late_failure"; }
  std::vector<Port> reads() const override { return {}; }
  std::vector<Port> writes() const override { return {}; }
  void snapshot() override {}
  void advance(Context&) override {
    if (fail_) throw std::runtime_error("injected late failure");
  }
  void commit() override {}
  void rollback() noexcept override {}
  void release_snapshot() noexcept override {}
  void set_fail(bool value) { fail_ = value; }

 private:
  bool fail_ = false;
};

struct Fixture {
  std::unique_ptr<Session> session;
  Publisher* publisher;
  Receiver* receiver;
  LateFailure* failure;
};

Fixture make_fixture(double at_end_first, double at_start_second) {
  auto session = std::make_unique<Session>(1000000);
  auto publisher = std::make_unique<Publisher>(at_end_first, at_start_second);
  auto receiver = std::make_unique<Receiver>();
  auto failure = std::make_unique<LateFailure>();
  auto* publisher_ptr = publisher.get();
  auto* receiver_ptr = receiver.get();
  auto* failure_ptr = failure.get();
  session->add(std::move(publisher));
  session->add(std::move(receiver));
  session->add(std::move(failure));
  session->compile();
  return {std::move(session), publisher_ptr, receiver_ptr, failure_ptr};
}

void check_two_ticks(double first, double second, bool retry, double expected) {
  auto fixture = make_fixture(first, second);
  fixture.session->tick();
  require(fixture.receiver->total() == first, "incorrect first tick");
  if (retry) {
    fixture.failure->set_fail(true);
    bool failed = false;
    try { fixture.session->tick(); }
    catch (const std::runtime_error& e) {
      failed = std::string(e.what()) == "injected late failure";
    }
    require(failed, "missing late failure");
    require(fixture.session->time_ns() == 1000000, "clock not restored");
    require(fixture.publisher->calls() == 1, "publisher not restored");
    require(fixture.receiver->total() == first, "receiver not restored");
    require(fixture.session->context().events().size() == (first ? 1u : 0u),
            "event history not restored");
    fixture.failure->set_fail(false);
  }
  fixture.session->tick();
  require(fixture.receiver->total() == expected, "event double counted or lost");
  require(fixture.session->time_ns() == 2000000, "final clock incorrect");
}

}  // namespace

int main() {
  try {
    check_two_ticks(1.0, 0.0, false, 1.0);  // End boundary is old next tick.
    check_two_ticks(1.0, 2.0, false, 3.0);  // Fresh start event is still read.
    check_two_ticks(1.0, 2.0, true, 3.0);   // Rollback and retry same start.
    check_two_ticks(0.0, 2.0, false, 2.0);  // New start with no prior event.
    auto interior = make_fixture(0.0, 0.0);
    interior.session->tick();
    require(interior.receiver->total() == 0.0, "empty tick changed receiver");
    std::cout << "PASS boundary_end=1 boundary_start=1 retry=1 empty=1\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL " << e.what() << '\n';
    return 1;
  }
}
